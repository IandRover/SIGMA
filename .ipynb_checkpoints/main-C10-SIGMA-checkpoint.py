import os
os.environ["WANDB_API_KEY"] = "dcf9600e0485401cbb0ddbb0f7be1c70f96b32ef"
os.environ["WANDB_MODE"] = "disabled"
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import ipdb
import wandb
from sigma_layer_vgg import SigmaLinear, SigmaConv, SigmaView
from utils import get_dataset, compute_SCL_loss, gradient_centralization
import datetime

import logging
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Create a logger instance for your module
info = logger.info  # Shorthand for logger.info


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# Training scheme group
method_parser = parser.add_argument_group("Method")
method_parser.add_argument('--method', type=str, default='SIGMA', choices=['SIGMA', 'BP', 'FA'])
method_parser.add_argument('--architecture', type=str, default='vgg', choices=['lenet', 'vgg'])
method_parser.add_argument('--actfunc', type=str, default='elu', choices=['tanh', 'elu', 'relu'])
# Dataset group
dataset_parser = parser.add_argument_group('Dataset')
dataset_parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['MNIST', 'CIFAR10'])
dataset_parser.add_argument('--batchsize', type=int, default=128)
# Training group # LR, optimizer, weight_decay, momentum
training_parser = parser.add_argument_group('Training')
training_parser.add_argument('--epochs', type=int, default=100)
training_parser.add_argument('--lr_bp', type=float, default=0.03)
training_parser.add_argument('--lr_F', type=float, default=0.1)
training_parser.add_argument('--lr_B', type=float, default=0.5)
training_parser.add_argument('--Fiter', type=int, default=0)
training_parser.add_argument('--Biter', type=int, default=0)
training_parser.add_argument('--optimizer', type=str, default='SGD', choices=['RMSprop', 'Adam', 'SGD'])
training_parser.add_argument('--GradC', type=int, default=0)
# Seed group
seed_parser = parser.add_argument_group('Seed')
seed_parser.add_argument('--seed', type=int, default=2023)
args, _ = parser.parse_known_args()

info("Parsed Arguments:")
for arg in vars(args):
    info(f"  {arg}: {getattr(args, arg)}")

# Set run_name
if args.method == "SIGMA":
    run_name = f"{args.dataset}_{args.method}_vgg_act{args.actfunc}_{args.optimizer}_F{args.lr_F}B{args.lr_B}_{args.seed}"
elif args.method == "BP":
    run_name = f"{args.dataset}_{args.method}_vgg_act{args.actfunc}_{args.optimizer}_{args.lr_bp}_{args.seed}"
time_stamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")

# Set wandb
wandb.init(
    project="opt-sigma",
    name=run_name,
    # track hyperparameters and run metadata
    config={
    "algorithm": args.method,
    "architecture": "VGG",
    "dataset": args.dataset,
    "epochs": args.epochs,
    "lr1": args.lr_bp if args.method == "BP" else args.lr_F,
    "lr2": args.lr_B if args.method == "SIGMA" else 0,
    "optimizer": args.optimizer,
    "seed": args.seed,
    "actfunc": args.actfunc,
    }
)

info(f"Run name: {run_name}")

# Set seed
torch.manual_seed(args.seed), np.random.seed(args.seed), torch.cuda.manual_seed(args.seed)

# Set device
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
info(f"Device: {device}")

# Get dataset
info(f"\n Loading {args.dataset} dataset")
train_loader, val_loader, test_loader = get_dataset(args)

class SigmaModel_SimpleCNN(nn.Module):
    def __init__(self, args):
        super(SigmaModel_SimpleCNN, self).__init__()
        if args.dataset == "CIFAR10":
            self.conv1 = SigmaConv(args, 3, 128, 3, 1, 1, 2, 2, 0)
            self.conv2 = SigmaConv(args, 128, 128, 3, 1, 1, 2, 2, 0)
            self.conv3 = SigmaConv(args, 128, 256, 3, 1, 1, 2, 2, 0)
            self.conv4 = SigmaConv(args, 256, 256, 3, 1, 1, 2, 2, 0)
            self.conv5 = SigmaConv(args, 256, 512, 3, 1, 1, 2, 2, 0)
            self.view1 = SigmaView((512, 1, 1), 512 * 1 * 1)
            self.fc1 = SigmaLinear(args, 512 * 1 * 1, 10)
                
        self.forward_params = list()
        self.backward_params = list()
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1]:
            forward_params, backward_params = layer.get_parameters()
            self.forward_params += forward_params
            self.backward_params += backward_params

    def forward(self, x, detach_grad=False, return_activations=True):   
        a1 = self.conv1(x, detach_grad)
        a2 = self.conv2(a1, detach_grad)
        a3 = self.conv3(a2, detach_grad)
        a4 = self.conv4(a3, detach_grad)
        a5 = self.conv5(a4, detach_grad)
        a5 = self.view1(a5, detach_grad)
        a6 = self.fc1(a5, detach_grad)
        return [a1, a2, a3, a4, a5, a6]

        
    def reverse(self, target, detach_grad=True, return_activations=True):
        if target.shape == torch.Size([10]): 
            target = F.one_hot(target, num_classes=10).float().to(target.device)
        b5 = self.fc1.reverse(target, detach_grad)
        # info(f"b5: {b5.shape}")
        b5 = self.view1.reverse(b5, detach_grad)
        # info(f"b5: {b5.shape}")
        b4 = self.conv5.reverse(b5, detach_grad)
        # info(f"b4: {b4.shape}")
        b3 = self.conv4.reverse(b4, detach_grad)
        # info(f"b3: {b3.shape}") 
        b2 = self.conv3.reverse(b3, detach_grad)
        # info(f"b2: {b2.shape}")
        b1 = self.conv2.reverse(b2, detach_grad)
        # info(f"b1: {b1.shape}")
        return [b1, b2, b3, b4, b5, target]

class SigmaLoss(nn.Module):
    def __init__(self, args):
        super(SigmaLoss, self).__init__()
        self.args = args
        self.final_criteria = nn.CrossEntropyLoss()
        self.local_criteria = compute_SCL_loss
        self.method = args.method
        
    def forward(self, activations, signals, target, method="final"):
        if method == "local":
            loss = list()
            for idx, (act, sig) in enumerate(zip(activations[:-1], signals[:-1])):
                if len(act.shape) == 4 and len(sig.shape) == 2: sig = sig.view(sig.shape[0], sig.shape[1], act.shape[2], act.shape[3]) 
                if len(act.shape) == 2 and len(sig.shape) == 4: act = act.view(act.shape[0], act.shape[1], sig.shape[2], sig.shape[3])
                loss += [self.local_criteria(act, sig, target)]
                # if idx in [0,1,2]:
                #     loss += [compute_SSL(sig, method="channel-based") * 0.1]
            loss += [self.final_criteria(activations[-1], target)]
            return sum(loss), loss[-1].item()
        elif method == "final":
            loss = self.final_criteria(activations[-1], target)
            return loss, loss.item()
        
def normalize_along_axis(x):
    x = x.reshape(len(x), -1)
    norm = torch.norm(x, dim=1, keepdim=True)
    return x / (norm + 1e-8)

def standardize_along_axis(x):
    x = x.reshape(len(x), -1)
    norm = torch.std(x, dim=1, keepdim=True)
    return x / (norm + 1e-8)

def compute_SSL(A, method=None):
    # Given input matrix A, compute the self-supervised loss
    if method == "class-based":
        assert A.shape[0] == 10
        A = A.reshape(10, -1)
        A_norm = normalize_along_axis(A)
        C = A_norm@A_norm.T
        identity = torch.eye(10).to(A.device)
        return F.mse_loss(C, identity)
    elif method == "channel-based":
        A = A.transpose(0, 1)
        assert A.shape[1] == 10
        A = A.reshape(A.shape[0], -1)
        A_norm = normalize_along_axis(A)
        C = A_norm@A_norm.T
        identity = torch.eye(A.shape[0]).to(A.device) * 1.1 - 0.1
        return F.mse_loss(C, identity)
    elif method == "class-channel-based":
        A = A.reshape(A.shape[0]*A.shape[1], -1)
        A_norm = normalize_along_axis(A)
        C = A_norm@A_norm.T
        identity = torch.eye(A.shape[0]).to(A.device)
        return F.mse_loss(C, identity)
    else:
        raise ValueError("Invalid method")
        
model = SigmaModel_SimpleCNN(args)
model.to(device)
if args.optimizer == "SGD" and args.method == "BP": 
    forward_optimizer = optim.SGD(model.forward_params, lr=args.lr_bp, momentum=0.9, weight_decay=0.0001)
    forward_scheduler = CosineAnnealingLR(forward_optimizer, T_max=args.epochs, eta_min=1e-5)
elif args.optimizer == "SGD" and args.method == "SIGMA":
    forward_optimizer = optim.SGD(model.forward_params, lr=args.lr_F, momentum=0.9, weight_decay=0.0001)
    forward_scheduler = CosineAnnealingLR(forward_optimizer, T_max=args.epochs, eta_min=1e-5)
    backward_optimizer = optim.SGD(model.backward_params, lr=args.lr_B, momentum=0.9, weight_decay=0.0001)
    backward_scheduler = CosineAnnealingLR(backward_optimizer, T_max=args.epochs, eta_min=1e-5)
criteria = SigmaLoss(args)
    
with torch.no_grad():
    signals = model.reverse(torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long().to(device), return_activations=True)

best_val_loss = float('inf')
info(f"\n  Start training for {args.epochs} epochs")
for epoch in range(args.epochs):
    train_loss, train_counter = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.method == "SIGMA":
            # if args.Fiter > 0:
            #     for i in range(args.Fiter):
            #         activations = model(data.to(device), detach_grad=True)
            #         with torch.no_grad():
            #             signals = model.reverse(torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long().to(device), detach_grad=True)
            #         loss, loss_item = criteria(activations, signals, target.to(device), method="local")
            #         loss *= loss * 0.1
            #         forward_optimizer.zero_grad(), loss.backward(), forward_optimizer.step()
            # if args.Biter > 0:
            #     for i in range(args.Biter):
            #         with torch.no_grad():
            #             activations = model(data.to(device), detach_grad=True)
            #         signals = model.reverse(torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long().to(device), detach_grad=True)
            #         loss, loss_item = criteria(activations, signals, target.to(device), method="local")
            #         loss *= loss * 0.1
            #         backward_optimizer.zero_grad(), loss.backward(), backward_optimizer.step()
            activations = model(data.to(device), detach_grad=True)
            signals = model.reverse(torch.Tensor([0,1,2,3,4,5,6,7,8,9]).long().to(device), detach_grad=True)
            if args.GradC: gradient_centralization(model)
            loss, loss_item = criteria(activations, signals, target.to(device), method="local")
            forward_optimizer.zero_grad(), backward_optimizer.zero_grad(), loss.backward()
            
            forward_optimizer.step(), backward_optimizer.step()
            train_loss += loss_item * len(data)
            train_counter += len(data)
        elif args.method == "BP":
            activations = model(data.to(device), detach_grad=False)
            loss, loss_item = criteria(activations, signals, target.to(device), method="final")
            forward_optimizer.zero_grad(), loss.backward(), forward_optimizer.step()
            train_loss += loss_item * len(data)
            train_counter += len(data)

    forward_scheduler.step()
    if args.method == "SIGMA": backward_scheduler.step()
    wandb.log({'train_loss': train_loss / train_counter}, step=epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx > 100: continue
        activations = model(data.to(device), detach_grad=True)
        loss, loss_item = criteria(activations, signals, target.to(device), method="final")
        loss *= loss * 0.1
        forward_optimizer.zero_grad(), loss.backward(), forward_optimizer.step()

    # Validation
    val_correct, val_loss, val_counter = 0, 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            val_counter += len(data)
            if args.method == "SIGMA":
                activations = model(data.to(device), detach_grad=True)
                _, loss_item = criteria(activations, signals, target.to(device), method="local")
            elif args.method == "BP":
                activations = model(data.to(device), detach_grad=True)
                _, loss_item = criteria(activations, signals, target.to(device), method="final")
            prediction = activations[-1].detach()
            _, predicted = torch.max(prediction, 1)
            val_correct += (predicted == target.to(device)).sum().item()
            val_loss += loss_item * len(data)

    wandb.log({'val_loss': val_loss / val_counter, 
               'val_acc': val_correct / val_counter, 
               }, step=epoch)
    
    info(f"""Epoch {epoch} | train loss {train_loss / train_counter:.4f} | val loss {val_loss / val_counter:.4f} | val acc {100 * val_correct / val_counter:.4f}""")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), f'./saved_models/{run_name}-{time_stamp}.pt')
        

# Eval on Test Set by loading the best model 
model.load_state_dict(torch.load(f'./saved_models/{run_name}-{time_stamp}.pt'))
model.eval()
correct, total = 0, 0
test_loss, test_counter = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data.to(device), detach_grad=False)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == target.to(device)).sum().item()
        loss, loss_item = criteria(activations, signals, target.to(device), method="final")
        test_loss += loss_item * len(data)
        test_counter += len(data)

wandb.log({'test_loss': test_loss / test_counter,
           'test_acc': 100 * correct / test_counter})

info(f'Epoch: {epoch}, Test Accuracy: {100 * correct / test_counter:.4f}%')