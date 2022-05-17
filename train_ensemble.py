
"""
Code adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Kaggles/DiabeticRetinopathy
Trains the ensemble of EfficientNet models (2xB3, 2xB4, 2xB5) using MSE loss with significant data augmentation
"""

import torch
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset
from torchvision.utils import save_image
from torchsummary import summary
from utils import (
    B3Config,
    B4Config,
    B5Config,
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    make_prediction,
    get_csv_for_blend,
)
from os import listdir
from os.path import isfile, join
import pandas as pd

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):
        # save examples and make sure they look ok with the data augmentation,
        # tip is to first set mean=[0,0,0], std=[1,1,1] so they look "normal"
        # save_image(data, f"images/hi_{batch_idx}.png")

        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.unsqueeze(1).float())

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"Loss average over epoch: {sum(losses)/len(losses)}")


def main():
    model_checkpoint_path = 'C:/GitHub/chrisnielsen-ophthalmology/diabetic-retinopathy-classification/models'
    results_output_path = 'C:/GitHub/chrisnielsen-ophthalmology/diabetic-retinopathy-classification/results'
    existing_result_files = [f for f in listdir(results_output_path) if isfile(join(results_output_path, f))]


    model_config_dict = {'b3': B3Config(),
                         'b4': B4Config(),
                         'b5': B5Config()}


    for model_type in ['b3','b4','b5']:
        print('Starting training for: ', model_type)
        model_config = model_config_dict[model_type]

        for ensemble_iter in range(model_config.number_of_ensemble_iters):
            model_output_log_name = model_type + '_' + str(ensemble_iter) + '.csv'
            if model_output_log_name in existing_result_files:
                if config.OVERWRITE_MODELS == False:
                    continue



            print('init datasets')
            train_ds = DRDataset(
                images_folder="C:/Data/Kaggle EyePACS/train_images_resized_512/",
                path_to_csv="C:/Data/Kaggle EyePACS/trainLabels.csv",
                transform=model_config.train_transforms,
            )

            val_ds = DRDataset(
                images_folder="C:/Data/Kaggle EyePACS/test_images_resized_512/",
                path_to_csv="C:/Data/Kaggle EyePACS/test_public.csv",
                transform=model_config.val_transforms,
            )


            print('init dataloaders')

            train_loader = DataLoader(
                train_ds,
                batch_size=model_config.batch_size,
                shuffle=True,
                num_workers=12, persistent_workers=True
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=model_config.batch_size,
                shuffle=False,
                num_workers=12, persistent_workers=True
            )

            print('init model')
            loss_fn = nn.MSELoss()

            print('loading', model_config.model_name)
            model = EfficientNet.from_pretrained(model_config.model_name)
            print(model)
            model._fc = nn.Linear(model_config.fc_size, 1)
            model = model.to(config.DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            scaler = torch.cuda.amp.GradScaler()


            results_dict = {'accuracy': [],
                            'kappa': []}
            for epoch in range(model_config.num_epochs):
                print('starting training', epoch)

                train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

                # get on validation
                preds, labels, accuracy = check_accuracy(val_loader, model, config.DEVICE)
                print(accuracy)
                # print(labels)
                # print(preds)

                print(f"QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights='quadratic')}")

                results_dict['accuracy'].append(accuracy)
                results_dict['kappa'].append(cohen_kappa_score(labels, preds, weights='quadratic'))

                # get on train
                #preds, labels = check_accuracy(train_loader, model, config.DEVICE)
                #print(f"QuadraticWeightedKappa (Training): {cohen_kappa_score(labels, preds, weights='quadratic')}")

                if config.SAVE_MODEL:

                    pd.DataFrame(results_dict).to_csv(results_output_path + '/' + model_type + '_' + str(ensemble_iter) + '.csv', index=False)

                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, filename=model_checkpoint_path + '/' + model_type + '_' + str(ensemble_iter) + '_' + str(epoch) + ".pth.tar")



if __name__ == "__main__":
    main()
