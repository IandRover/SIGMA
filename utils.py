import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn.functional as F

import torch
import copy
import wandb


def get_dataset(args):
    # split train into train and validation
    # may use different transformation function for train, valid, and test
    if args.dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset_raw = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_dataset, val_dataset = random_split(train_dataset_raw, [0.8, 0.2])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
        return train_loader, val_loader, test_loader
    
    if args.dataset == 'CIFAR10':
        train_transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        # transforms.Normalize([0, 0, 0], [1, 1, 1]),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        test_transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
        return train_loader, val_loader, test_loader
    

def gradient_centralization(model):
    with torch.no_grad():
        for p1, p2 in model.named_parameters():
            if "bias" in p1 or p2.grad is None: continue
            if len(p2.shape) == 2: p2.grad -= p2.grad.mean(dim=1,keepdim=True)
            elif len(p2.shape) == 4: p2.grad -= p2.grad.mean(dim=[1,2,3],keepdim=True) 

def normalize_along_axis(x):
    x = x.reshape(len(x), -1)
    norm = torch.norm(x, dim=1, keepdim=True)
    return x / (norm + 1e-8)

def get_activation_function(args):
    if args.actfunc == "tanh": return F.tanh
    elif args.actfunc == "elu": return F.elu
    elif args.actfunc == "relu": return F.relu
    else: raise ValueError(f"Activation function {args.actfunc} not implemented")
    
def compute_SCL_loss(A, B, target):
    A_norm, B_norm = normalize_along_axis(A), normalize_along_axis(B)
    C = A_norm@B_norm.T
    if len(B) == 10:
        target_A = F.one_hot(target, num_classes=10).float().to(target.device)
        target_B = F.one_hot(torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long(), num_classes=10).float().to(target.device)
        identity = target_A@target_B.T
    else:
        target_A = F.one_hot(target, num_classes=10).float().to(target.device)
        identity = target_A@target_A.T
    C
    return F.mse_loss(C, identity)

def compute_SCL_loss(A, B, target, error_mask=None):

    if error_mask is not None:
        A = A[error_mask]
        target = target[error_mask]

    A_norm, B_norm = normalize_along_axis(A), normalize_along_axis(B)
    C = A_norm@B_norm.T
    if len(B) == 10:
        target_A = F.one_hot(target, num_classes=10).float().to(target.device)
        target_B = F.one_hot(torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long(), num_classes=10).float().to(target.device)
        identity = target_A@target_B.T
    else:
        target_A = F.one_hot(target, num_classes=10).float().to(target.device)
        identity = target_A@target_A.T
    return F.mse_loss(C, identity)

# def compute_SCL_loss_square(A, B, target):
#     A_norm, B_norm = normalize_along_axis(A), normalize_along_axis(B)
#     C = A_norm@B_norm.T
#     target_A = F.one_hot(target, num_classes=10).float().to(target.device)
#     # target_B = F.one_hot(torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long(), num_classes=10).float().to(target.device)
#     identity = target_A@target_A.T
#     return F.mse_loss(C, identity)