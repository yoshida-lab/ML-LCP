import numpy as np
import random
import os


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import itertools
from itertools import cycle
#import torchinfo

import copy

import pandas as pd

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True





def set_device(key=None):
    if key=='cpu':
        MyDevice = torch.device('cpu')
    else:
        MyDevice = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(MyDevice)
    return MyDevice






def fix_seed(seed=373):
    # Numpy
    np.random.seed(seed)
    # for built-in random
    random.seed(int(seed))
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
    
    
    
## Model training

class MyDataset(Dataset):
    def __init__(self, df, IDs, labels):
        self.df = df
        self.IDs = IDs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.df[idx], self.IDs[idx], self.labels[idx]
    
    

def fun_CreateDataset(x, fml, y, device):
    if y is not None:
        if isinstance(y, pd.Series) or (isinstance(y, pd.DataFrame) and y.shape[1] == 1):
            return (
            torch.tensor(x.loc[fml].values, dtype=torch.float32, device=device),
            torch.tensor(fml.index.values.reshape(-1, 1), dtype=torch.float32, device=device),
            torch.tensor(y.loc[fml].values.reshape(-1,1), dtype=torch.float32, device=device)
        )
        else:
            return (
            torch.tensor(x.loc[fml].values, dtype=torch.float32, device=device),
            torch.tensor(fml.index.values.reshape(-1, 1), dtype=torch.float32, device=device),
            torch.tensor(y.loc[fml].values, dtype=torch.float32, device=device)
        )
    if y is None:
        return (
            torch.tensor(x.loc[fml].values, dtype=torch.float32, device=device),
            torch.tensor(fml.index.values.reshape(-1, 1), dtype=torch.float32, device=device)
        )




    
    
    
    
class MyModel(nn.Module):
    def __init__(self, input_dim, layers_dict):
        super(MyModel, self).__init__()

        layers = {}
        for name, output_dim in layers_dict.items():
            layers[name] = nn.Linear(input_dim, output_dim)
            input_dim = output_dim
        self.layers = nn.ModuleDict(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers.values()):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x
    
    def initialize_weights(self, layer_names=None, seed=373):
        torch.manual_seed(seed)
        
        for name, layer in self.layers.items():
            if layer_names is None or name in layer_names:
                # nn.init.xavier_uniform_(layer.weight)
                nn.init.kaiming_uniform_(layer.weight)
                # nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
                # nn.init.constant_(layer.bias, 1)
                # nn.init.uniform_(layer.bias, -0.01, 0.01)
                
                
class MyModel_Mulit(nn.Module):
    def __init__(self, input_dim, layers_dict):
        super(MyModel, self).__init__()

        layers = {}
        for name, output_dim in layers_dict.items():
            layers[name] = nn.Linear(input_dim, output_dim)
            input_dim = output_dim
        self.layers = nn.ModuleDict(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers.values()):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x
    
    '''
    def initialize_weights(self, layer_names=None, seed=373):
        torch.manual_seed(seed)
        
        for name, layer in self.layers.items():
            if layer_names is None or name in layer_names:
                # nn.init.xavier_uniform_(layer.weight)
                nn.init.kaiming_uniform_(layer.weight)
                # nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
                # nn.init.constant_(layer.bias, 1)
                # nn.init.uniform_(layer.bias, -0.01, 0.01)
    '''
                    
                    
                    
                    
        
    
    
    


def fun_TrainModel(
    model, 
    dataloader_train, 
    dataset_val, 
    learning_rate, 
    n_epochs, 
    patience, 
    loss_th, 
    opt, 
    seed,
    device,
    weight_decay=1e-4,
    amsgrad=False,
    lr_decay=1e-6,
    schedule=False
):

    fix_seed(seed)
    
    if not amsgrad:
        if isinstance(learning_rate, float):
            optimizer = opt(model.parameters(), lr=learning_rate, weight_decay=weight_decay, lr_decay=lr_decay)
            # optimizer = opt(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        if isinstance(learning_rate, dict):
            params_update = []
            for name, param in model.named_parameters():
                if learning_rate[name.split('.')[1]]==0:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                params_update.append({'params': [param], 'lr': learning_rate[name.split('.')[1]]})

            optimizer = opt(params_update, weight_decay=weight_decay, lr_decay=lr_decay)
            # optimizer = opt(params_update, weight_decay=weight_decay, momentum=0.9)
    else:
        if isinstance(learning_rate, float):
            optimizer = opt(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=amsgrad)
        if isinstance(learning_rate, dict):
            params_update = []
            for name, param in model.named_parameters():
                if learning_rate[name.split('.')[1]]==0:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                params_update.append({'params': [param], 'lr': learning_rate[name.split('.')[1]]})

            optimizer = opt(params_update, weight_decay=weight_decay, amsgrad=amsgrad)
            
#     if isinstance(learning_rate, float):
#         optimizer = opt(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=amsgrad)
#     if isinstance(learning_rate, dict):
#         params_update = []
#         for name, param in model.named_parameters():
#             if learning_rate[name.split('.')[1]]==0:
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
#             params_update.append({'params': [param], 'lr': learning_rate[name.split('.')[1]]})

#         optimizer = opt(params_update, weight_decay=weight_decay, amsgrad=amsgrad)

    
    
    best_loss = None
    epochs_no_improve = 0
    criterion = nn.MSELoss()

    ## Training
    train_losses = []
    val_losses = []
    train_len = len(dataloader_train)
    
    if schedule:
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


    # fix_seed(n_seed)
    for epoch in range(n_epochs):
        train_loss = 0
        val_loss = 0


        # Train loop
        model.train()
        for (data, df_ID, label) in dataloader_train:

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, label)
            train_loss += loss.item()  # accumulate the losses

            loss.backward()
            optimizer.step()
            
           


        # Validation loop
        model.eval()
        with torch.no_grad():
            
            data, data_ID, label = dataset_val
            
            output = model(data)
            
            val_loss += criterion(output, label).item()
            
    
        # if schedule:
        #     scheduler.step()
            # scheduler.step(val_loss)

            
            
        ## Compute the average losses
        train_loss /= len(dataloader_train)
        train_losses.append(train_loss)
        val_losses.append(val_loss)


        if (epoch == 0) or ((epoch+1) % 10 == 0):
            print(f'Epoch {epoch+1}: train loss = {train_loss}, val loss = {val_loss}')



        if patience is not None:
            if loss_th is None or min(train_losses) < loss_th:  ## Warm-up
                # Early stopping
                if best_loss is None:
                    best_loss = val_loss
                    best_weights = copy.deepcopy(model.state_dict())
                elif val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    best_weights = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        # clear_output()
                        print('Early stopping!')
                        print('======================')
                        break
                        
                ## LR scheduler
                if schedule:
                    # scheduler.step()
                    scheduler.step(val_loss)
                    
        else:
                ## LR scheduler
                if schedule:
                    # scheduler.step()
                    scheduler.step(val_loss)
            

    if epoch==(n_epochs-1):
        # clear_output()
        print('Reach max epochs')
        print('======================')
        
    if patience is not None:
        model.load_state_dict(best_weights)
        
    return model, train_losses, val_losses


def fun_MakePrediction(model, dataset, indices, labels_exist, device):
    if labels_exist==True:
        data, df_ID, label = dataset
        output = model(data)

        return pd.DataFrame({
            'Scaled Observation' : label.detach().cpu().numpy().reshape(-1).tolist(),
            'Scaled Prediction' : output.detach().cpu().numpy().reshape(-1).tolist()
        }, index = indices)
    
    
    if labels_exist==False:
        data, df_ID = dataset
        output = model(data)

        return pd.DataFrame({
            'Scaled Prediction' : output.detach().cpu().numpy().reshape(-1).tolist()
        }, index = indices)
        
        
