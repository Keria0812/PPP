import torch
from .predictor_modules import *


class Encoder(nn.Module):
    def __init__(self, dim=256, layers=6, heads=8, dropout=0.1):
        super(Encoder, self).__init__()
        self._lane_len = 50
        self._lane_feature = 7
        self._crosswalk_len = 30
        self._crosswalk_feature = 3
        self.agent_encoder = AgentEncoder(history_channel=11)
        self.lane_encoder = MapEncoder(self._lane_feature, self._lane_len)
        self.self_attention = SelfTransformer()
        self.crosswalk_encoder = MapEncoder(self._crosswalk_feature, self._crosswalk_len)
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
        self.current_mlp = nn.Sequential(
            nn.Linear(self._lane_feature, int(dim/4)),
            nn.ReLU(),
            nn.Linear(int(dim/4), dim),
        )

    def forward(self, inputs):
        # agents
        ego = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']
        encoded_neighbors = self.agent_encoder(neighbors)
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)

        ego_current = ego[:, -1, :]
        ego_current_feature = self.current_mlp(ego_current)

        encoded_ego_current = ego_current_feature.unsqueeze(1)


        map_lanes = inputs['lanes']
        map_crosswalks = inputs['crosswalks']
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        mask = torch.cat([actors_mask,lanes_mask,crosswalks_mask],dim=1)

        actor_attention = self.fusion_encoder(torch.cat([encoded_ego_current,encoded_neighbors],dim=1),src_key_padding_mask=actors_mask)
        map_attention = self.fusion_encoder(torch.cat([encoded_map_lanes,encoded_map_crosswalks], dim=1),src_key_padding_mask=torch.cat([lanes_mask,crosswalks_mask],dim=1))
        encoding = torch.cat([actor_attention, map_attention],dim=1)

        # outputs
        encoder_outputs = {
            'actors': actors,
            'encoding': encoding,
            'mask': mask,
            'actors_mask':actors_mask,
            'map_mask': torch.cat([lanes_mask,crosswalks_mask],dim=1),
            'route_lanes': inputs['route_lanes'],
            'ref_paths': inputs['ref_paths'],
            'ego_current_encoding':encoded_ego_current
        }

        return encoder_outputs

class Decoder(nn.Module):
    def __init__(self, 
                 neighbors=20, 
                 modalities=6, 
                 dropout=0.1,
                 dim=256,
                 num_heads=8,
                 ):
        super(Decoder, self).__init__()
        self._future_len=80
        self.neighbors = neighbors
        self.embed_dims= 256
        self.cross_attention = CrossTransformer()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.attention = MultiModalTransformer()
        self.ref_encoder = Ref_Encoder(dim=5)
        self.decoder = GMM()
        self.decoder_layer = DecoderLayer(dim, num_heads, dropout)
        self.m_pos = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.normal_(self.m_pos, mean=0.0, std=0.01)
        self.future_encoder = FutureEncoder()
        self.plan_decoder = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), 
                                          nn.Linear(512, 256), nn.ELU(), nn.Dropout(0.1), nn.Linear(256, self._future_len*2))
        self.interaction_stage = Interaction(modalities)

    def forward(self, encoder_outputs):
        decoder_outputs = {}
        current_states = encoder_outputs['actors'][:, :, -1]
        encoding, mask = encoder_outputs['encoding'], encoder_outputs['mask']

        encoding_agent = encoding[:,:self.neighbors+1,:]
        encoding_map = encoding[:, self.neighbors+1:, :]

        al_interaction = self.cross_attention(encoding_agent,encoding_map,encoding_map, mask=encoder_outputs['map_mask'])
        #print(al_interaction.size())#torch.Size([32, 21, 256])
        aa_interaction = self.cross_attention(encoding_agent,encoding_agent,encoding_agent, mask=encoder_outputs['actors_mask'])
        #print(aa_interaction.size())#torch.Size([32, 21, 256])
        interaction = torch.cat([al_interaction,aa_interaction], dim=-1)
        interaction_mlp = self.fusion_mlp(interaction)
        
        initial_attention, score = self.attention(interaction_mlp,al_interaction,al_interaction,mask = encoder_outputs['actors_mask'])
        
        for k in range(3):
            initial_attention, score = self.interaction_stage(self.neighbors, score, initial_attention, encoding, mask)
        agents_pred, scores = self.decoder(initial_attention,current_states)

        ref_paths = encoder_outputs['ref_paths']
        ego_features = []
        for sample_idx in range(encoding_agent.size(0)):
            sample_ref_paths =  ref_paths[sample_idx][:,:,:5].unsqueeze(0)

            sample_agent_pred = agents_pred[sample_idx].unsqueeze(0)

            sample_scores = scores[sample_idx].unsqueeze(0)

            sample_agent_pred_future = self.future_encoder(sample_agent_pred[..., :2],current_states[sample_idx].unsqueeze(0)[:, :self.neighbors+1])
            futures = (sample_agent_pred_future * sample_scores.softmax(-1).unsqueeze(-1)).mean(dim=2)  

            env = torch.cat([futures,encoding[sample_idx].unsqueeze(0)], dim=1)
            env_mask = torch.cat([encoder_outputs['actors_mask'][sample_idx].unsqueeze(0) ,mask[sample_idx].unsqueeze(0)], dim=1)

            ref_mask = torch.zeros((sample_ref_paths.size(0), sample_ref_paths.size(1)), dtype=torch.bool).to('cuda')
            for i in range(sample_ref_paths.size(1)):
                if torch.all(sample_ref_paths[0, i] == 0):
                    ref_mask[0, i] = True

            encoded_ref_paths= torch.stack([self.ref_encoder(sample_ref_paths[:, i]) for i in range(sample_ref_paths.shape[1])],dim=1).unsqueeze(2)

            for _ in range(4):
                encoded_ref_paths, score = self.decoder_layer(
                    encoded_ref_paths,
                    env,
                    ref_mask=ref_mask,
                    env_mask=env_mask,
                    memory_position=self.m_pos,
                )

            encoded_ref_paths = encoded_ref_paths.squeeze(2)

            score = score.squeeze(2).squeeze(2)

            score_softmax = score.softmax(-1)
            _, max_score_index = torch.max(score_softmax, dim=-1)
            sample_final_ego_feature = encoded_ref_paths[torch.arange(encoded_ref_paths.size(0)), max_score_index]

            ego_features.append(sample_final_ego_feature)
        ego_features = torch.cat(ego_features,dim=0)

        ego_control = self.plan_decoder(ego_features)
        ego_control = ego_control.reshape(-1, self._future_len, 2)
        initial_state = encoder_outputs['actors'][:, 0, -1]
        ego_plan = self.compute_dynamics(ego_control, initial_state)
        
        decoder_outputs = {
            'agents_pred':agents_pred,
            'scores':scores,
            'ego_plan':ego_plan
        }

        return decoder_outputs
    
    def compute_dynamics(self, controls, initial_conditions):
        time_step = 0.1
        max_acceleration = 5
        max_yaw_rate = 0.5
        
        initial_velocity = torch.hypot(initial_conditions[..., 3], initial_conditions[..., 4])
        
        velocity = initial_velocity[:, None] + torch.cumsum(controls[..., 0].clamp(-max_acceleration, max_acceleration) * time_step, dim=-1)
        velocity = torch.clamp(velocity, min=0)


        yaw_rate = controls[..., 1].clamp(-max_yaw_rate, max_yaw_rate) * velocity
        yaw_angle = initial_conditions[:, None, 2] + torch.cumsum(yaw_rate * time_step, dim=-1)
        yaw_angle = torch.fmod(yaw_angle, 2 * torch.pi)

        velocity_x = velocity * torch.cos(yaw_angle)
        velocity_y = velocity * torch.sin(yaw_angle)

        x_position = initial_conditions[:, None, 0] + torch.cumsum(velocity_x * time_step, dim=-1)
        y_position = initial_conditions[:, None, 1] + torch.cumsum(velocity_y * time_step, dim=-1)

        return torch.stack((x_position, y_position, yaw_angle), dim=-1)
    
class PPP(nn.Module):
    def __init__(self, neighbors=20):
        super(PPP, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(neighbors)

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs= self.decoder(encoder_outputs)
        ego_plan = decoder_outputs['ego_plan']

        return decoder_outputs, ego_plan