import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from SwinLSC_model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    # Set device to GPU if available, otherwise CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create weights directory if it doesn't exist
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # Initialize TensorBoard writer for logging
    tb_writer = SummaryWriter()

    # Read and split the dataset into training and validation sets
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # Define image size and data transformations for training and validation
    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    # Initialize training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Initialize validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # Set batch size and number of workers for data loading
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    
    # Create data loader for training set
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    # Create data loader for validation set
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # Create model and move it to the specified device
    model = create_model(num_classes=args.num_classes).to(device)

    # Load pre-trained weights if specified
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # Model structure is weight name (key), weight value (value)
        # Remove weights related to classification categories
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
                # This code is used to load pre-trained model weights to a new model, will prompt which weights are missing

    # Freeze layers if specified
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # Freeze all weights except for the head
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # Get parameters that require gradients
    pg = [p for p in model.parameters() if p.requires_grad]
    # Initialize optimizer with AdamW
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Multiply learning rate by 0.1 every epoch
    
    # Lists to store training and validation metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc, train_macro_f1, train_recall, train_fps = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate on validation set
        val_loss, val_acc, valid_macro_f1, valid_recall, valid_fps = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Log metrics to TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        scheduler.step()  # Update learning rate

        # Save model at specific epochs
        if epoch == 24 or epoch == 49 or epoch == 99 or epoch == 149:
            print('train_losse', train_losses)
            print('train_accs', train_accs)
            print('val_losse', val_losses)
            print('val_accs', val_accs)
            torch.save(model.state_dict(), "./my_weights2/model-{}.pth".format(epoch))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)

    # Root directory of the dataset
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="Your image directory location")

    # Path to pre-trained weights, if not loading, set to empty string
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # Whether to freeze weights
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
