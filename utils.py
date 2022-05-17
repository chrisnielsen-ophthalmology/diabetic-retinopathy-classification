

"""
Code adapted from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Kaggles/DiabeticRetinopathy
Defines a bunch of helper functions and classes which are used to assist the training process
"""


import torch
import pandas as pd
import numpy as np
import config
from tqdm import tqdm
import warnings
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2


augmentation_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.CLAHE(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=8, max_width=8, p=0.3),
        A.ShiftScaleRotate(),
        A.ElasticTransform(),
        # A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
        A.Normalize(
            mean=[0.3766757268814456, 0.26143159198778143, 0.1866619521640538],
            std=[0.2920528728656642, 0.21087333272470993, 0.17061016606220822],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

class B3Config:
    def __init__(self):
        self.number_of_ensemble_iters = 2
        self.model_name = "efficientnet-b3"
        self.fc_size = 1536
        self.batch_size = 24
        self.num_epochs = 30
        self.resize_width = 512

        self.train_transforms = A.Compose(
            [
                augmentation_transforms
            ]
        )

        self.val_transforms = A.Compose(
            [
                A.Resize(height=self.resize_width , width=self.resize_width ),
                A.Normalize(
                    mean=[0.3766757268814456, 0.26143159198778143, 0.1866619521640538],
                    std=[0.2920528728656642, 0.21087333272470993, 0.17061016606220822],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )




class B4Config:
    def __init__(self):
        self.number_of_ensemble_iters = 2
        self.model_name = "efficientnet-b4"
        self.fc_size = 1792
        self.batch_size = 24
        self.num_epochs = 30
        self.resize_width = 512

        self.train_transforms = A.Compose(
            [
                augmentation_transforms
            ]
        )

        self.val_transforms = A.Compose(
            [
                A.Resize(height=self.resize_width, width=self.resize_width),
                A.Normalize(
                    mean=[0.3766757268814456, 0.26143159198778143, 0.1866619521640538],
                    std=[0.2920528728656642, 0.21087333272470993, 0.17061016606220822],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

class B5Config:
    def __init__(self):
        self.number_of_ensemble_iters = 2
        self.model_name = "efficientnet-b5"
        self.fc_size = 2048
        self.batch_size = 16
        self.num_epochs = 20
        self.resize_width = 512

        self.train_transforms = A.Compose(
            [
                augmentation_transforms
            ]
        )

        self.val_transforms = A.Compose(
            [
                A.Resize(height=self.resize_width, width=self.resize_width),
                A.Normalize(
                    mean=[0.3766757268814456, 0.26143159198778143, 0.1866619521640538],
                    std=[0.2920528728656642, 0.21087333272470993, 0.17061016606220822],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )




def make_prediction(model, loader, output_csv="submission.csv"):
    preds = []
    filenames = []
    model.eval()
    all_preds, all_labels = [], []

    for x, y, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            predictions = model(x)
            # Convert MSE floats to integer predictions
            predictions[predictions < 0.5] = 0
            predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
            predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
            predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
            predictions[(predictions >= 3.5) & (predictions < 10000000)] = 4
            predictions = predictions.long().squeeze(1)
            preds.append(predictions.cpu().numpy())

            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            filenames += files

    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    model.train()
    print("Done with predictions")
    return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(all_labels, axis=0, dtype=np.int64)


def check_accuracy(loader, model, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0

    for x, y, filename in tqdm(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            predictions = model(x)

        # Convert MSE floats to integer predictions
        predictions[predictions < 0.5] = 0
        predictions[(predictions >= 0.5) & (predictions < 1.5)] = 1
        predictions[(predictions >= 1.5) & (predictions < 2.5)] = 2
        predictions[(predictions >= 2.5) & (predictions < 3.5)] = 3
        predictions[(predictions >= 3.5) & (predictions < 100)] = 4
        predictions = predictions.long().view(-1)
        y = y.view(-1)

        num_correct += (predictions == y).sum()
        num_samples += predictions.shape[0]

        # add to lists
        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())


    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )
    model.train()
    return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(all_labels, axis=0, dtype=np.int64), float(num_correct)/float(num_samples)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    #optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_csv_for_blend(loader, model, output_csv_file):
    warnings.warn("Important to have shuffle=False (and to ensure batch size is even size) when running get_csv_for_blend also set val_transforms to train_loader!")
    model.eval()
    filename_first = []
    filename_second = []
    labels_first = []
    labels_second = []
    all_features = []

    for idx, (images, y, image_files) in enumerate(tqdm(loader)):
        images = images.to(config.DEVICE)

        with torch.no_grad():
            features = F.adaptive_avg_pool2d(
                model.extract_features(images), output_size=1
            )
            features_logits = features.reshape(features.shape[0] // 2, 2, features.shape[1])
            preds = model(images).reshape(images.shape[0] // 2, 2, 1)
            new_features = (
                torch.cat([features_logits, preds], dim=2)
                .view(preds.shape[0], -1)
                .cpu()
                .numpy()
            )
            all_features.append(new_features)
            filename_first += image_files[::2]
            filename_second += image_files[1::2]
            labels_first.append(y[::2].cpu().numpy())
            labels_second.append(y[1::2].cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    df = pd.DataFrame(
        data=all_features, columns=[f"f_{idx}" for idx in range(all_features.shape[1])]
    )
    df["label_first"] = np.concatenate(labels_first, axis=0)
    df["label_second"] = np.concatenate(labels_second, axis=0)
    df["file_first"] = filename_first
    df["file_second"] = filename_second
    df.to_csv(output_csv_file, index=False)
    model.train()
