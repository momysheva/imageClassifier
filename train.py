import torch
from torchvision import transforms, datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser (description = 'Parser _ training script')

parser.add_argument ('data_dir', help = 'Input data directory. Mandatory', type=str)
parser.add_argument ('--save_dir', help = 'Input saving directory. Optional',  type=str, default='.')
parser.add_argument ('--arch', help = 'Default is VGG-16-BN, otherwise input VGG13 or AlexNet', type=str, default='VGG-16-BN')
parser.add_argument ('--learning_r', help = 'Learning rate - default is 0.001', type = float, default = 0.001)
parser.add_argument ('--hidden_units', help = 'Hidden units. Default val 2048', type = int, default = 4096)
parser.add_argument ('--epochs', help = 'Epochs as integer - default is 5', type = int, default = 3)
parser.add_argument ('--gpu', type=str2bool, nargs='?', const=True, default=False, help="Use GPU or not")
parser.add_argument ('--batch_size', help = "Input batch size", type = int, default = 128)

args = parser.parse_args()




def train_step(
        model: torch.nn.Module,
        train_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        device: PyTorch device to use for training.

    Returns:
        Average loss for the epoch.
    """

    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predictions = torch.max(output, 1)

        correct_counts = predictions.eq(target.data.view_as(predictions))

        accuracy = torch.mean(correct_counts.type(torch.FloatTensor))

        train_acc += accuracy.item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc

def val_step(
        model: torch.nn.Module,
        val_loader,
        loss_fn: torch.nn.Module,
        device: torch.device,
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss and accuracy for the val set.
    """

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            val_loss += loss_fn(output, target).item()

            _, predictions = torch.max(output, 1)

            correct_counts = predictions.eq(target.data.view_as(predictions))

            accuracy = torch.mean(correct_counts.type(torch.FloatTensor))

            val_acc += accuracy.item()


    val_loss /= len(val_loader)

    val_acc /= len(val_loader)
    return val_loss, val_acc


def trainer(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        save_dir: str,
):
    """
    Train and evaluate model.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        lr_scheduler: PyTorch learning rate scheduler.
        device: PyTorch device to use for training.
        epochs: Number of epochs to train the model for.

    Returns:
        Average loss and accuracy for the val set.
    """

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = 1e10

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}:")
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)

        val_loss, val_acc = val_step(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print()

        
        
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))


    return results


data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


data_train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
data_valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
data_test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_train_datasets = datasets.ImageFolder(train_dir,transform=data_train_transforms)
image_valid_datasets = datasets.ImageFolder(valid_dir,transform=data_valid_transforms)
image_test_datasets = datasets.ImageFolder(test_dir,transform=data_test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(image_train_datasets,batch_size=args.batch_size,shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(image_valid_datasets,batch_size=args.batch_size)
test_dataloaders = torch.utils.data.DataLoader(image_test_datasets,batch_size=args.batch_size)



if args.arch == 'VGG-16-BN':
    model = models.vgg16_bn(weights=True)
    no_input_features = 25088
elif args.arch == 'VGG13':
    model = models.vgg13(weights=True)
    no_input_features = 25088
elif args.arch == 'AlexNet':
    model = models.alexnet(weights=True)
    no_input_features = 9216


classifier = nn.Sequential(
    nn.Linear(in_features=no_input_features, out_features=args.hidden_units),  # Linear layer
    nn.ReLU(),  # ReLU activation
    nn.Dropout(0.05),
    nn.Linear(in_features=args.hidden_units, out_features=102),  # Another linear layer
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

if args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.classifier.parameters(), lr=args.learning_r)

epochs = args.epochs

print(f"Training started for {epochs} epochs")
print("Model: ", args.arch)
print("Learning rate: ", args.learning_r, end="\n\n")

results = trainer(model, train_dataloaders, valid_dataloaders, criterion, optimizer, device, epochs, save_dir=args.save_dir)

print("Training completed")

model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_model.pt")))

#Save the checkpoint
model.class_to_idx = image_train_datasets.class_to_idx

checkpoint = {'arch': args.arch,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, os.path.join(args.save_dir, "checkpoint.pth"))


print("Checkpoint saved")








