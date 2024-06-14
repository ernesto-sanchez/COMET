import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="DiffPO",
    
)

import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=500,
    foldername="",
    propnet = None
):
    # Control random seed in the current script.
    torch.manual_seed(0)
    np.random.seed(0)
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p0 = int(0.25 * config["epochs"])
    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    p3 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )
    history = {'train_loss':[], 'val_rmse':[]}
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model.forward(batch = train_batch, is_train=1, propnet = propnet)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        wandb.log({"train loss":avg_loss / batch_no})

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            print("Start validation")
            model.eval()
            # some initial settings
            val_nsample = 15

            pehe_val = AverageMeter()
            y0_val = AverageMeter()
            y1_val = AverageMeter()

            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        output = model.evaluate(valid_batch, val_nsample)
                        (
                            samples,
                            c_target,
                            eval_points,
                            observed_points,
                            observed_time,
                        ) = output
                        samples = samples.permute(0, 1, 3, 2)  
                        c_target = c_target.permute(0, 2, 1)  
                        eval_points = eval_points.permute(0, 2, 1)
                        observed_points = observed_points.permute(0, 2, 1)
                        samples_mean = samples.mean(dim=1)
                        obs_data = torch.squeeze(c_target) 
                        true_ite = obs_data[:, 3] - obs_data[:, 4] 
                        est_data = torch.squeeze(samples_mean.values) 
                        pred_y0 = est_data[:, 1]
                        pred_y1 = est_data[:, 2]
                        est_ite = pred_y0 - pred_y1
                        diff_ite = np.mean((true_ite.cpu().numpy()-est_ite.cpu().numpy())**2)
                        pehe_val.update(diff_ite, obs_data.size(0))    
                    pehe = np.sqrt(pehe_val.avg)
                    print("PEHE VAL = {:0.3g}".format(pehe))
    wandb.finish()

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    # Control random seed in the current script.
    torch.manual_seed(0)
    np.random.seed(0)

    with torch.no_grad():
        model.eval()
        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        pehe_test = AverageMeter()
        y0_test = AverageMeter()
        y1_test = AverageMeter()

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample) 
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  
                c_target = c_target.permute(0, 2, 1)  
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_mean = samples.mean(dim=1) 
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)
                obs_data = torch.squeeze(c_target, 2) 
                true_ite = obs_data[:, 3] - obs_data[:, 4] # (8,)
                est_data = torch.squeeze(samples_mean.values) 
                pred_y0 = est_data[:, 1]
                pred_y1 = est_data[:, 2]
                est_ite = pred_y0 - pred_y1
                diff_ite = np.mean((true_ite.cpu().numpy()-est_ite.cpu().numpy())**2)
                pehe_test.update(diff_ite, obs_data.size(0))    
            pehe = np.sqrt(pehe_test.avg)
            print("PEHE TEST = {:0.3g}".format(pehe))


