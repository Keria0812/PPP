import os
import csv
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from PPP.predictor import PPP
from torch.utils.data import DataLoader
from PPP.train_utils import *



def train_epoch(data_loader, model, optimizer):
    epoch_loss = []
    epoch_metrics = []
    model.train()
    
    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            ego_agent_past = torch.stack([nf for nf in batch['ego_agent_past']]).to(args.device)
            neighbor_agents_past = torch.stack([nf for nf in batch['neighbor_agents_past']]).to(args.device)
            route_lanes = torch.stack([nf for nf in batch['route_lanes']]).to(args.device)
            map_lanes = torch.stack([nf for nf in batch['map_lanes']]).to(args.device)
            map_crosswalks = torch.stack([nf for nf in batch['map_crosswalks']]).to(args.device)      

            inputs = {
                'ego_agent_past': ego_agent_past,
                'neighbor_agents_past': neighbor_agents_past,
                'route_lanes': route_lanes,
                'lanes': map_lanes,
                'crosswalks': map_crosswalks,
                'ref_paths': batch['ref_paths']
            }


            ego_future = torch.stack([nf for nf in batch['ego_agent_future']]).to(args.device)
            neighbors_future = batch['neighbor_agents_future']
            neighbors_future = torch.stack([nf for nf in neighbors_future]).to(args.device)
            
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            # call the mdoel
            optimizer.zero_grad()
            decoder_outputs, ego_plan = model(inputs)
            loss: torch.tensor = 0
            gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)


            trajectories = decoder_outputs['agents_pred']
            scores = decoder_outputs['scores']
            predictions = trajectories[:, 1:] * neighbors_future_valid[:, :, None, :, 0, None]
            
            plan = trajectories[:, :1]
            trajectories = torch.cat([plan, predictions], dim=1)

            nll_loss, results = NLL_Loss(trajectories, gt_future)

            B, N = trajectories.shape[0], trajectories.shape[1]
            distances = torch.norm(trajectories[..., :2] - gt_future[:, :, None, :, :2], dim=-1)
            best_mode = torch.argmin(distances.mean(dim=-1), dim=-1)
            cross_entropy_loss = Cross_Entropy_Loss(scores, best_mode, gt_future)

            loss += nll_loss
            loss += cross_entropy_loss


            prediction = results[:, 1:]
            l1loss = l1_loss(ego_plan, ego_future)

            l2loss = l2_loss(ego_plan, ego_future) *0.2

            loss += l1loss
            loss += l2loss 
            # loss backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # compute metrics
            metrics = motion_metrics(ego_plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    planningAHE, planningFHE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
    logging.info(f"plannerADE: {planningADE:.4f}, plannerFDE: {planningFDE:.4f}, " +
                 f"plannerAHE: {planningAHE:.4f}, plannerFHE: {planningFHE:.4f}, " +
                 f"predictorADE: {predictionADE:.4f}, predictorFDE: {predictionFDE:.4f}\n")
        
    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, model):
    epoch_loss = []
    epoch_metrics = []
    model.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            # prepare data
            
            ego_agent_past = torch.stack([nf for nf in batch['ego_agent_past']]).to(args.device)
            neighbor_agents_past = torch.stack([nf for nf in batch['neighbor_agents_past']]).to(args.device)
            route_lanes = torch.stack([nf for nf in batch['route_lanes']]).to(args.device)
            map_lanes = torch.stack([nf for nf in batch['map_lanes']]).to(args.device)
            map_crosswalks = torch.stack([nf for nf in batch['map_crosswalks']]).to(args.device)      

            inputs = {
                'ego_agent_past': ego_agent_past,
                'neighbor_agents_past': neighbor_agents_past,
                'route_lanes': route_lanes,
                'lanes': map_lanes,
                'crosswalks': map_crosswalks,
                'ref_paths': batch['ref_paths']
            }
            ego_future = torch.stack([nf for nf in batch['ego_agent_future']]).to(args.device)
            neighbors_future = batch['neighbor_agents_future']
            neighbors_future = torch.stack([nf for nf in neighbors_future]).to(args.device)
            neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

            with torch.no_grad():
                decoder_outputs, ego_plan = model(inputs)
                loss: torch.tensor = 0
                gt_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)


                trajectories = decoder_outputs['agents_pred']
                scores = decoder_outputs['scores']
                predictions = trajectories[:, 1:] * neighbors_future_valid[:, :, None, :, 0, None]
                plan = trajectories[:, :1]
                trajectories = torch.cat([plan, predictions], dim=1)

                nll_loss, results = NLL_Loss(trajectories, gt_future)

                B, N = trajectories.shape[0], trajectories.shape[1]
                distances = torch.norm(trajectories[..., :2] - gt_future[:, :, None, :, :2], dim=-1)
                best_mode = torch.argmin(distances.mean(dim=-1), dim=-1)
                cross_entropy_loss = Cross_Entropy_Loss(scores, best_mode, gt_future)

                loss += nll_loss
                loss += cross_entropy_loss


                prediction = results[:, 1:]
                l1loss = l1_loss(ego_plan, ego_future)

                l2loss = l2_loss(ego_plan, ego_future) *0.2
                loss += l1loss
                loss += l2loss 

            # compute metrics
            metrics = motion_metrics(ego_plan, prediction, ego_future, neighbors_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    planningAHE, planningFHE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 4]), np.mean(epoch_metrics[:, 5])
    epoch_metrics = [planningADE, planningFDE, planningAHE, planningFHE, predictionADE, predictionFDE]
    logging.info(f"val-plannerADE: {planningADE:.4f}, val-plannerFDE: {planningFDE:.4f}, " +
                 f"val-plannerAHE: {planningAHE:.4f}, val-plannerFHE: {planningFHE:.4f}, " +
                 f"val-predictorADE: {predictionADE:.4f}, val-predictorFDE: {predictionFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics

def collate_fn(batch):
    batch = from_numpy(batch)

    
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]

    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data).to(args.device)

    return data


def model_training():
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up model
    PPP_model = PPP()
    PPP_model = PPP_model.to(args.device)
    logging.info("Model Params: {}".format(sum(p.numel() for p in PPP_model.parameters())))

    # set up optimizer
    optimizer = optim.AdamW(PPP_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(args.train_set + '/*.npz', args.num_neighbors)
    valid_set = DrivingData(args.valid_set + '/*.npz', args.num_neighbors)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    
    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        train_loss, train_metrics = train_epoch(train_loader, PPP_model, optimizer)
        val_loss, val_metrics = valid_epoch(valid_loader, PPP_model)

        # save to training log
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
               'train-planningADE': train_metrics[0], 'train-planningFDE': train_metrics[1], 
               'train-planningAHE': train_metrics[2], 'train-planningFHE': train_metrics[3], 
               'train-predictionADE': train_metrics[4], 'train-predictionFDE': train_metrics[5],
               'val-planningADE': val_metrics[0], 'val-planningFDE': val_metrics[1], 
               'val-planningAHE': val_metrics[2], 'val-planningFHE': val_metrics[3],
               'val-predictionADE': val_metrics[4], 'val-predictionFDE': val_metrics[5]}

        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        torch.save(PPP_model.state_dict(), f'training_log/{args.name}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="PPP Decision-making")
    parser.add_argument('--train_set', type=str, help='path to train data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=20)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=128)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    # Run
    model_training()