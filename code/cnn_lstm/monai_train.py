# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import ImageDataset
from monai.transforms import (
    AddChannel,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    EnsureType,
)


from config import params, split_train_val


def get_class(score):
    """
    Categorize each score into one of the five classes (bins)
    Returns values from 0-5 (6 classes)
    Classes: (0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)
    """
    if score < 10:
        return 0
    elif score >= 10 and score < 20:
        return 1
    elif score >= 20 and score < 40:
        return 2
    elif score >= 40 and score < 60:
        return 3
    elif score >= 60 and score < 80:
        return 4
    else:
        return 5


def get_img_labels(subs):
    root_dir = "/data/fmri/data"
    imgs, labels = [], []
    for sub in subs:
        sub_path = os.path.join(root_dir, sub, f"{sub}.preproc")
        labels_0_back = os.path.join(root_dir, sub, "0back_VAS-f.1D")
        labels_2_back = os.path.join(root_dir, sub, "2back_VAS-f.1D")
        for img_name in os.listdir(sub_path):
            labels_path = labels_0_back if "0back" in img_name else labels_2_back
            with open(labels_path, "r") as labels_file:
                curr_labels = labels_file.readlines()
            curr_labels = [int(value.replace("\n", "")) for value in curr_labels]
            sess_id = int(img_name.split("back.")[1][2])
            try:
                actual_label = curr_labels[int(sess_id)]
            except IndexError:
                continue
            img_path = os.path.join(sub_path, img_name)
            imgs.append(img_path)
            labels.append(get_class(actual_label))
    return imgs, labels


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # IXI dataset as a demo, downloadable from https://brain-development.org/ixi-dataset/
    # the path of ixi IXI-T1 dataset

    # Define transforms
    train_transforms = Compose(
        [
            ScaleIntensity(),
            AddChannel(),
            Resize((96, 96, 96)),
            RandRotate90(),
            EnsureType(),
        ]
    )
    val_transforms = Compose(
        [ScaleIntensity(), AddChannel(), Resize((96, 96, 96)), EnsureType()]
    )

    # Prepare dataset to load
    train_subs, val_subs = split_train_val(val_pct=0.2)
    print(f"Train: {train_subs}\nValidation: {val_subs}")

    train_imgs, train_labels = get_img_labels(train_subs)
    val_imgs, val_labels = get_img_labels(val_subs)

    # print(f'Train images: {train_imgs}\n\Train Labels: {train_labels}')

    # Create a training data loader
    train_ds = ImageDataset(
        image_files=train_imgs, labels=train_labels, transform=train_transforms
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_ds = ImageDataset(
        image_files=val_imgs, val=val_labels, transform=val_transforms
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, num_workers=2, pin_memory=torch.cuda.is_available()
    )

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(
        spatial_dims=3, in_channels=1, out_channels=2
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(5):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{5}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                num_correct = 0.0
                metric_count = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(
                        device
                    )
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                    metric_count += len(value)
                    num_correct += value.sum().item()
                metric = num_correct / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        "best_metric_model_classification3d_array.pth",
                    )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", metric, epoch + 1)
    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    writer.close()


if __name__ == "__main__":
    main()
