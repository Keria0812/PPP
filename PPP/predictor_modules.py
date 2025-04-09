import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import *
from typing import Optional
from torch import Tensor
import logging
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    

class Ref_Encoder(nn.Module):
    def __init__(self, dim):
        super(Ref_Encoder, self).__init__()
        self.motion = nn.LSTM(dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs)
        output = traj[:, -1]

        return output

class AgentEncoder(nn.Module):
    def __init__(
        self,
        state_channel=6,
        history_channel=11,
        dim=256,
        hist_steps=21,
        drop_path=0.2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.state_channel = state_channel
        self.hist_steps = hist_steps


        self.history_encoder = NATSequenceEncoder(
            in_chans=history_channel, embed_dim=dim // 4, drop_path_rate=drop_path
        )




    def forward(self, inputs):

        #print(f"agent_feature: {agent_feature.size()}")#agent_feature: torch.Size([24, 50, 20, 9])
        agent_feature = inputs
        valid_mask = ~((agent_feature[..., 0] == 0) & (agent_feature[..., 1] == 0) & (agent_feature[..., 2] == 0))
        bs, A, T, _ = agent_feature.shape
        agent_feature = agent_feature.view(bs * A, T, -1)
        valid_agent_mask = valid_mask.any(-1).flatten()

        
        #print(f"agent_feature: {agent_feature.size()}")#([1200, 20, 9])
        if not valid_agent_mask.any():
            logging.info(f"no_neighbor")
            return torch.zeros(bs, A, self.dim, device=agent_feature.device)

        x_agent_tmp = self.history_encoder(
            agent_feature[valid_agent_mask].permute(0, 2, 1).contiguous()
        )
        #print(x_agent_tmp.size())
        x_agent = torch.zeros(bs * A, self.dim, device=agent_feature.device)
        x_agent[valid_agent_mask] = x_agent_tmp
        x_agent = x_agent.view(bs, A, self.dim)
        #print(x_agent.size())#torch.Size([4, 20, 128])
        '''
        if not self.use_ego_history:
            ego_feature = data["current_state"][:, : self.state_channel]
            x_ego = self.ego_state_emb(ego_feature)
            x_agent[:, 0] = x_ego
        '''
  

        return x_agent 






class MapEncoder(nn.Module):
    def __init__(self, map_dim, map_len):
        super(MapEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(map_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.position_encode = PositionalEncoding(max_len=map_len)

    def segment_map(self, map, map_encoding):
        B, N_e, N_p, D = map_encoding.shape 
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask

    def forward(self, input):
        output = self.position_encode(self.point_net(input))
        encoding, mask = self.segment_map(input, output)

        return encoding, mask
    

class FutureEncoder(nn.Module):
    def __init__(self):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 256))

    def state_process(self, trajs, current_states):
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-6)).unsqueeze(-1)
        trajs = torch.cat([trajs, theta, v], dim=-1) # (x, y, heading, vx, vy)

        return trajs

    def forward(self, trajs, current_states):
        trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs.detach())
        output = torch.max(trajs, dim=-2).values

        return output


class GMM(nn.Module):
    def __init__(self, modalities=6):
        super(GMM, self).__init__()
        self.modalities = modalities
        self._future_len = 80
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, 1))
    
    def forward(self, input,current_states):
        B, N, M, _ = input.shape
        traj = self.gaussian(input).view(B, N, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        traj[..., :2] += current_states[:, :N, None, None, :2]
        score = self.score(input).squeeze(-1)

        return traj, score




class SelfTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossTransformer(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output





class Interaction(nn.Module):
    def __init__(self, modalities, dim=256):
        super(Interaction, self).__init__()
        self.modalities = modalities
        self._agents = 21
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(self._agents, dim)
        self.interaction_encoder = SelfTransformer()
        self.query_encoder = CrossTransformer()
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, 1))
        self.register_buffer('modal', torch.arange(modalities).long())
        self.register_buffer('agent', torch.arange(self._agents).long())


    def forward(self, neighbor, scores, last_content, encoding, mask):
        N = neighbor+1

        futures = (last_content * scores.softmax(-1).unsqueeze(-1)).mean(dim=2)    

        interaction = self.interaction_encoder(futures, mask[:, :N])
        
        encoding = torch.cat([interaction, encoding], dim=1)

        mask = torch.cat([mask[:, :N], mask], dim=1)
        mask = mask.unsqueeze(1).expand(-1, N, -1).clone()
        for i in range(N):
            mask[:, i, i] = 1

        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent)
        query = last_content
        query = encoding[:, :N, None] + multi_modal_query[None, :, :] + agent_query[:, None, :]
        query_content = torch.stack([self.query_encoder(query[:, i], encoding, encoding, mask[:, i]) for i in range(N)], dim=1)

        scores = self.score(last_content).squeeze(-1)#torch.Size([32, 21, 6])



        return query_content, scores
    

class MultiModalTransformer(nn.Module):
    def __init__(self, modes=6, output_dim=256):
        super(MultiModalTransformer, self).__init__()
        self.modes = modes
        self.attention = nn.ModuleList([nn.MultiheadAttention(256, 4, 0.1, batch_first=True) for _ in range(modes)])
        self.ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, output_dim), nn.LayerNorm(output_dim))
        self.score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, query, key, value, mask=None):
        attention_output = []
        for i in range(self.modes):
            attention_output.append(self.attention[i](query, key, value, key_padding_mask=mask)[0])
        attention_output = torch.stack(attention_output, dim=1)
        output = self.ffn(attention_output).transpose(1,2)
        query = query.unsqueeze(2).expand(-1,-1, 6, -1).clone()
        score = self.score(query).squeeze(-1)

        return output,score

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Self-attention layers
        self.self_to_self_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.local_to_local_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout_rate, batch_first=True
        )

        # Feed-forward network
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

        # Scoring layer
        self.score_layer = nn.Sequential(
            nn.Linear(embedding_dim, 64), 
            nn.ELU(), 
            nn.Linear(64, 1)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(
        self,
        encoded_ref_paths,
        env,
        ref_mask: Optional[Tensor] = None,
        env_mask: Optional[Tensor] = None,
        memory_position: Optional[Tensor] = None,
    ):
        """
        Forward pass through the decoder layer.
        
        encoded_ref_paths: (batch_size, num, pad, embedding_dim)
        ref_mask: (batch_size, num)
        env_mask: (batch_size, num_memory)
        """
        batch_size, num, pad, embedding_dim = encoded_ref_paths.shape

        # Reshaping and applying the first attention mechanism
        encoded_ref_paths = encoded_ref_paths.transpose(1, 2).reshape(batch_size * pad, num, embedding_dim)
        attention_input = self.norm1(encoded_ref_paths)
        attention_output = self.self_to_self_attention(
            attention_input, attention_input, attention_input, key_padding_mask=ref_mask.repeat(pad, 1)
        )[0]
        encoded_ref_paths = encoded_ref_paths + self.dropout1(attention_output)

        # Process valid entries using the second attention mechanism
        reshaped_tensor = encoded_ref_paths.reshape(batch_size, pad, num, embedding_dim).transpose(1, 2).reshape(batch_size * num, pad, embedding_dim)
        valid_mask = ~ref_mask.reshape(-1)
        valid_tensor = reshaped_tensor[valid_mask]
        valid_tensor = self.norm2(valid_tensor)
        local_attention_output, _ = self.local_to_local_attention(valid_tensor + memory_position, valid_tensor + memory_position, valid_tensor)
        valid_tensor = valid_tensor + self.dropout2(local_attention_output)
        encoded_ref_paths = torch.zeros_like(reshaped_tensor)
        encoded_ref_paths[valid_mask] = valid_tensor

        # Cross-attention mechanism with memory tensor
        encoded_ref_paths = encoded_ref_paths.reshape(batch_size, num, pad, embedding_dim).view(batch_size, num * pad, embedding_dim)
        encoded_ref_paths = self.norm3(encoded_ref_paths)
        cross_attention_output = self.cross_attention(
            encoded_ref_paths, env, env, key_padding_mask=env_mask
        )[0]
        encoded_ref_paths = encoded_ref_paths + self.dropout2(cross_attention_output)

        # Feed-forward network
        encoded_ref_paths = self.norm4(encoded_ref_paths)
        encoded_ref_paths = self.feed_forward_network(encoded_ref_paths)
        encoded_ref_paths = encoded_ref_paths + self.dropout3(encoded_ref_paths)
        encoded_ref_paths = encoded_ref_paths.reshape(batch_size, num, pad, embedding_dim)

        # Scoring
        score_output = self.score_layer(encoded_ref_paths)

        return encoded_ref_paths, score_output
