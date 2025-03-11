import os
import sys
import json
import pickle
import random
import time
from sklearn.metrics import f1_score, recall_score

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    """
    Read and split the dataset into training and validation sets.
    
    Args:
        root: Root directory of the dataset
        val_rate: Proportion of validation set
        
    Returns:
        train_images_path: List of paths to training images
        train_images_label: List of labels for training images
        val_images_path: List of paths to validation images
        val_images_label: List of labels for validation images
    """
    random.seed(0)  # Ensure reproducibility of random results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Traverse folders, each folder corresponds to a category
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sort to ensure consistency across platforms
    flower_class.sort()
    # Generate category names and corresponding numerical indices
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # Store all image paths in the training set
    train_images_label = []  # Store index information corresponding to training set images
    val_images_path = []  # Store all image paths in the validation set
    val_images_label = []  # Store index information corresponding to validation set images
    every_class_num = []  # Store the total number of samples for each category
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # Supported file extensions
    # Traverse files in each folder
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # Get all file paths supported by traversing
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # Sort to ensure consistency across platforms
        images.sort()
        # Get the index corresponding to this category
        image_class = class_indices[cla]
        # Record the number of samples in this category
        every_class_num.append(len(images))
        # Randomly sample validation samples by proportion
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # If the path is in the sampled validation set, store it in the validation set
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # Otherwise store in training set
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # Draw bar chart of the number of each category
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # Replace horizontal coordinates 0,1,2,3,4 with corresponding category names
        plt.xticks(range(len(flower_class)), flower_class)
        # Add numerical labels on the bar chart
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # Set x-coordinate
        plt.xlabel('image class')
        # Set y-coordinate
        plt.ylabel('number of images')
        # Set the title of the bar chart
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    """
    Plot images from the data loader.
    
    Args:
        data_loader: Data loader containing images and labels
    """
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Reverse Normalize operation
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # Remove x-axis ticks
            plt.yticks([])  # Remove y-axis ticks
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    """
    Write list to pickle file.
    
    Args:
        list_info: List to be written
        file_name: Name of the pickle file
    """
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    """
    Read list from pickle file.
    
    Args:
        file_name: Name of the pickle file
        
    Returns:
        info_list: List read from the pickle file
    """
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to be trained
        optimizer: Optimizer for updating model parameters
        data_loader: Data loader for training data
        device: Device to run the model on
        epoch: Current epoch number
        
    Returns:
        accu_loss.item() / (step + 1): Average loss
        top1_acc: Top-1 accuracy
        macro_f1: Macro F1 score
        recall: Recall
        fps: Frames per second
    """
    print("Current learning rate:", optimizer.param_groups[0]['lr'])
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss
    accu_num = torch.zeros(1).to(device)   # Accumulated number of correctly predicted samples
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    # Additional variables
    all_preds = []
    all_labels = []
    start_time = time.time()
    total_images = 0

    for step, data in enumerate(data_loader):
        # label: tensor([1, 2, 3, 1, 1, 0, 2, 4])
        # images is usually a four-dimensional tensor with shape (B, C, H, W)
        images, labels = data
        sample_num += images.shape[0]
        total_images += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        # Collect prediction results and true labels
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate additional metrics
    top1_acc = accu_num.item() / sample_num
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    end_time = time.time()
    fps = total_images / (end_time - start_time)

    return (
        accu_loss.item() / (step + 1),  # Average loss
        top1_acc,                       # Top-1 accuracy
        macro_f1,                       # Macro F1 score
        recall,                         # Recall
        fps                             # Images processed per second
    )


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: Model to be evaluated
        data_loader: Data loader for validation data
        device: Device to run the model on
        epoch: Current epoch number
        
    Returns:
        accu_loss.item() / (step + 1): Average loss
        top1_acc: Top-1 accuracy
        macro_f1: Macro F1 score
        recall: Recall
        fps: Frames per second
    """
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Accumulated number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # Accumulated loss

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    all_preds = []
    all_labels = []
    start_time = time.time()
    total_images = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        total_images += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        # Collect prediction results and true labels
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate additional metrics
    top1_acc = accu_num.item() / sample_num
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    end_time = time.time()
    fps = total_images / (end_time - start_time)

    return (
        accu_loss.item() / (step + 1),  # Average loss
        top1_acc,                       # Top-1 accuracy
        macro_f1,                       # Macro F1 score
        recall,                         # Recall
        fps                             # Images processed per second
    )
