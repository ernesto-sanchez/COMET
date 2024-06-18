import pickle
import yaml
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def process_func(path: str, aug_rate=1, train=True, dataset_name = 'acic', current_id='0'):
    # data = pd.read_csv(path, sep = ',', decimal = ',', skiprows=[0])
    data_directory = os.path.dirname(os.path.dirname(path)) #this line is new
    data = pd.read_csv(path, sep = ',', decimal = ',')
    data.replace("?", np.nan, inplace=True)
    data_aug = pd.concat([data] * aug_rate)

    observed_values = data_aug.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.copy()

    if dataset_name == 'acic2016':
        load_mask_path = "./data_acic2016/acic2016_mask/" + current_id + ".csv"
        print(load_mask_path)


    # ========================
    # acic2018
    if dataset_name == 'acic2018':
        load_mask_path = os.path.join(data_directory,'masked', current_id + "_merged" "_masked" + ".csv")
        print(load_mask_path)

    load_mask = pd.read_csv(load_mask_path, sep = ',', decimal = ',')
    load_mask = load_mask.values.astype("float32")

    if train:
            
        gt_masks = load_mask

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype(int)
        gt_masks = gt_masks.astype(int)

    else:
        gt_masks = load_mask
        # no yf for testing
        gt_masks[:, 1] = 0 # mask y0
        gt_masks[:, 2] = 0 # mask y1
        

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype(int)
        gt_masks = gt_masks.astype(int)


    return observed_values, observed_masks, gt_masks


class acic_dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(
        self, eval_length=100, use_index_list=None, aug_rate=1, missing_ratio=0.1, seed=0, train=True, dataset_name = 'acic', current_id='0'
    ):  
        if dataset_name == 'acic2016':
            self.eval_length = 87
        if dataset_name == 'acic2018':
            # self.eval_length = 182
            #new code: gotta keep the sample_id_col out
            self.eval_length = 181

        np.random.seed(seed)

        if dataset_name == 'acic2016':

            dataset_path = "./data_acic2016/acic2016_norm_data/" + current_id + ".csv"

            print('dataset_path', dataset_path)

            processed_data_path = (
                f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}.pk"
            ) # modify the processed data path
            processed_data_path_norm = (
                f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
            )
            
            os.system('rm {}'.format(processed_data_path))

        # ========================
        # acic2018
        if dataset_name == 'acic2018':

            # Flag: new code
            project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.dirname(project_path)

            # dataset_path = "./data_acic2018/acic2018_norm_data/" + current_id + ".csv"
            dataset_path = os.path.join(data_path, "ACIC2018", "merged", current_id + "_merged" + ".csv")

            print('dataset_path', dataset_path)

            # processed_data_path = (
            #     f"./data_acic2018/missing_ratio-{missing_ratio}_seed-{seed}.pk"
            # ) # modify the processed data path
            # processed_data_path_norm = (
            #     f"./data_acic2018/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
            # )
            processed_data_path = os.path.join(data_path, "ACIC2018","processed", f"missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk")
            processed_data_path_norm = os.path.join(data_path, "ACIC2018","processed", f"missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk")

            
            os.system('rm {}'.format(processed_data_path))

        if not os.path.isfile(processed_data_path):
            # original code: missing_ratio argument not needed!
           # self.observed_values, self.observed_masks, self.gt_masks = process_func(
            #     dataset_path, aug_rate=aug_rate, missing_ratio=missing_ratio, train=train, dataset_name=dataset_name, current_id=current_id
            # )
            self.observed_values, self.observed_masks, self.gt_masks = process_func(
                dataset_path, aug_rate=aug_rate, train=train, dataset_name=dataset_name, current_id=current_id
            )

            #New code:
            #Create empty pickle file in processed_data_path. Otherwise get error Exception has occurred: FileNotFoundError in the next line

        
 
            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
            print("--------Normalized dataset loaded--------")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list


    def __getitem__(self, org_index):
        index = self.use_index_list[org_index] 
        s = {
            "observed_data": self.observed_values[index], 
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1, dataset_name = 'acic2018', current_id='0'):
    dataset = acic_dataset(missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, current_id=current_id)
    print(f"Dataset size:{len(dataset)} entries") #  Dataset size: 747 entries

    indlist = np.arange(len(dataset))

    tsi = int(len(dataset) * 0.8)
    print('test start index', tsi)
    if tsi % 8 == 1 or int(len(dataset) * 0.2) % 8 == 1:
        tsi = tsi + 3

    if dataset_name == 'acic2016':
        test_index = indlist[tsi:]
        remain_index = np.arange(0, tsi)

        np.random.shuffle(remain_index)
        num_train = (int)(len(remain_index) * 1)

        train_index = remain_index[: tsi] 
        valid_index = remain_index[: int(tsi*0.1)] 

        processed_data_path_norm = (
            f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}_current_id-{current_id}_max-min_norm.pk"
        )
        # if not os.path.isfile(processed_data_path_norm):
        # normalize anyway.
        print(
            "------------- Perform data normalization and store the mean value of each column.--------------"
        )

    if dataset_name == 'acic2018':
        test_index = indlist[tsi:]
        remain_index = np.arange(0, tsi)

        np.random.shuffle(remain_index)
        num_train = (int)(len(remain_index) * 1)

        train_index = remain_index[: tsi] 
        valid_index = remain_index[: int(tsi*0.1)] 

        # old code
        # processed_data_path_norm = (
        #     f"./data_acic2018/missing_ratio-{missing_ratio}_seed-{seed}_current_id-{current_id}_max-min_norm.pk"
        # )

        #new code
        
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.dirname(project_path)

        processed_data_path_norm = os.path.join(data_path, "ACIC2018","processed", f"missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk")

        print(
            "------------- Perform data normalization and store the mean value of each column.--------------"
        )
        # data transformation after train-test split.


    
    col_num = dataset.observed_values.shape[1]

    with open(processed_data_path_norm, "wb") as f:
        pickle.dump(
            [dataset.observed_values, dataset.observed_masks, dataset.gt_masks], f
        )

    # Create datasets and corresponding data loaders objects.
    train_dataset = acic_dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, current_id=current_id
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = acic_dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, train=False, dataset_name = dataset_name, current_id=current_id
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = acic_dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, train=False, dataset_name = dataset_name, current_id=current_id
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader
