import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn.functional as F

import torch
import copy
import wandb



def validate(model, val_loader, device, args, criteria, signals, epoch):

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

    return val_loss / val_counter, val_correct / val_counter

def save_best_model(model, val_loss, best_val_loss, best_model, run_name, time_stamp):

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), f'./saved_models/{run_name}-{time_stamp}.pt')

    return best_val_loss, best_model

def test(model, test_loader, device, args, criteria, signals):
    model.eval()
    correct, total = 0, 0
    test_loss, test_counter = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data.to(device), detach_grad=False)
            _, predicted = torch.max(outputs[-1].data, 1)
            correct += (predicted == target.to(device)).sum().item()
            loss, loss_item = criteria(outputs, 0, target.to(device), method="final")

            test_loss += loss_item * len(data)
            test_counter += len(data)

    return test_loss / test_counter, correct / test_counter