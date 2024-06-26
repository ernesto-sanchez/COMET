import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import wandb

wandb.init(project="DiffPO")

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
                # The forward method returns loss.

                ## print('propnet in train function', propnet)
                # loss = model(train_batch, propnet)
                loss = model.forward(batch = train_batch, is_train=1, propnet = propnet)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                wandb.log({"train_loss_diffusion": loss.item()})
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )


            lr_scheduler.step()


        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            print('Start validation!!!')
            print('Epoch:', epoch_no)

            model.eval()
            avg_loss_valid = 0
            # some initial settings
            val_nsample = 50
            val_scaler = 1
            mse_total = 0
            mae_total = 0
            evalpoints_total = 0

            pehe_val = AverageMeter()
            y0_val = AverageMeter()
            y1_val = AverageMeter()

            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        output = model.evaluate(valid_batch, val_nsample)
                        samples, observed_data, target_mask, observed_mask, observed_tp = output
                        samples_median = torch.median(samples, dim=1).values

                        obs_data = observed_data.squeeze(1)
                        # print(obs_data.shape)
                        true_ite = obs_data[:, 1] - obs_data[:, 2]
                       
                        est_data = samples_median

                        pred_y0 = est_data[:, 0]
                        pred_y1 = est_data[:, 1]

                        # check rmse potential outcome (with mu0, mu1)
                        diff_y0 = np.mean((pred_y0.cpu().numpy()-obs_data[:, 1].cpu().numpy())**2)
                        y0_val.update(diff_y0, obs_data.size(0))    
                        diff_y1 = np.mean((pred_y1.cpu().numpy()-obs_data[:, 2].cpu().numpy())**2)
                        y1_val.update(diff_y1, obs_data.size(0)) 

                        est_ite = pred_y0 - pred_y1
                        diff_ite = np.mean((true_ite.cpu().numpy()-est_ite.cpu().numpy())**2)
                        wandb.log({"val_pehe": np.sqrt(diff_ite)})
                        pehe_val.update(diff_ite, obs_data.size(0))    
                    
                    print('====================================')
                    print('##### End evaluation!!')

                    pehe = np.sqrt(pehe_val.avg)
                    print("PEHE VAL = {:0.3g}".format(pehe))


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    # Control random seed in the current script.
    torch.manual_seed(0)
    np.random.seed(0)

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        pehe_test = AverageMeter()
        y0_test = AverageMeter()
        y1_test = AverageMeter()

        # for uncertainty
        y0_samples = []
        y1_samples = []
        y0_true_list = []
        y1_true_list = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):

                output = model.evaluate(test_batch, nsample) 
                samples, observed_data, target_mask, observed_mask, observed_tp = output
                y0_samples.append(samples[:,:,0]) 
                y1_samples.append(samples[:,:,1]) 
                samples_median = torch.median(samples, dim=1).values
                obs_data = observed_data.squeeze(1)
                true_ite = obs_data[:, 1] - obs_data[:, 2]   
                est_data = samples_median
                pred_y0 = est_data[:, 0]
                pred_y1 = est_data[:, 1]
                diff_y0 = np.mean((pred_y0.cpu().numpy()-obs_data[:, 1].cpu().numpy())**2)
                y0_test.update(diff_y0, obs_data.size(0))    
                diff_y1 = np.mean((pred_y1.cpu().numpy()-obs_data[:, 2].cpu().numpy())**2)
                y1_test.update(diff_y1, obs_data.size(0)) 
                y0_true_list.append(obs_data[:, 1])
                y1_true_list.append(obs_data[:, 2])
                est_ite = pred_y0 - pred_y1
                diff_ite = np.mean((true_ite.cpu().numpy()-est_ite.cpu().numpy())**2)
                pehe_test.update(diff_ite, obs_data.size(0))    

            print('====================================')
            
            pehe = np.sqrt(pehe_test.avg)
            print("PEHE TEST = {:0.3g}".format(pehe))
            print('Finish test')

#---------------uncertainty estimation-------------------------
            pred_samples_y0 = torch.cat(y0_samples, dim=0)
            pred_samples_y1 = torch.cat(y1_samples, dim=0)

            truth_y0 = torch.cat(y0_true_list, dim=0) 
            truth_y1 = torch.cat(y1_true_list, dim=0) 
            prob_0, median_width_0 = compute_interval(pred_samples_y0, truth_y0)
            prob_1, median_width_1 = compute_interval(pred_samples_y1, truth_y1)
#----------------------------------------------------------------


def check_intervel(confidence_level, y_pred, y_true):
    lower = (1 - confidence_level) / 2
    upper = 1 - lower
    lower_quantile = torch.quantile(y_pred, lower)
    upper_quantile = torch.quantile(y_pred, upper)
    in_quantiles = torch.logical_and(y_true >= lower_quantile, y_true <= upper_quantile)
    return lower_quantile, upper_quantile, in_quantiles

def compute_interval(po_samples, y_true):
    counter = 0
    width_list = []
    for i in range(po_samples.shape[0]):
        lower_quantile, upper_quantile, in_quantiles = check_intervel(confidence_level=0.95, y_pred= po_samples[i, :], y_true=y_true[i])
        if in_quantiles == True:
            counter+=1
        width = upper_quantile - lower_quantile
        width_list.append(width.unsqueeze(0))
    prob = (counter/po_samples.shape[0])
    all_width = torch.cat(width_list, dim=0)
    median_width = torch.median(all_width, dim=0).values
    return prob, median_width