# %%
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
# import time
# from sympy.solvers import solve
# from sympy import Symbol
# from sympy import *
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# # **Inequality Finder**

# %%
x_points=[]
y_points=[]
lambdas=[]
def drawLine(x0,x1,y0,y1):
  x=[x0,x1]
  y=[y0,y1]
  plt.plot(x0, y0, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")#show line bettwen 2 ponits
  plt.plot(x,y)
  m=(y1-y0)/(x1-x0)
  return lambda x:y0+m*(x-x0)# return the line functionbettwen the 2 points




# def cut(func1,func2,current):
#   arr1=np.array([func1(x) for x in np.arange(0,max_distance,0.001)])
#   plt.plot(np.linspace(0,max_distance,len(arr1)),arr1)#plot the paralel line

#   x = Symbol('x', real=True)
#   x_l=solve(func1(x)-func2(x),x)# find cut
#   min=max_distance
#   for i in range(len(x_l)):
#     if x_l[i]<min and x_l[i]>current:# look for the closet cut to the right of the graph
#       min=x_l[i]
#   y_current=func1(min)# get y of the cut    
#   if min<max_distance:# we will ad the point if the cut doesnt pass the barier 
#     plt.plot(min, y_current, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#     x_points.append(min)
#     y_points.append(y_current)
#   return min


# def findInequality(L,P,Q,I,Z):
#   global  x_points
#   global  y_points
#   x_points=[]
#   y_points=[]

#   derivative=lambda x:-(L*P*Q*(x+I)**(P-1))/(Q*(x+I)**P+Z)**2+(L*P*Q*(x-I)**(P-1))/(Q*(x-I)**P+Z)**2+1
#   funcatin=lambda x:x + (L / (Q * (x + I) ** P + Z)) - (L / (Q * (x - I) ** P + Z))
#   x0=start
#   arr2=np.array([funcatin(x) for x in np.arange(0,max_distance,0.001)])#dynamic function


#   while x0<max_distance:# if the new point is bigger then max, just return 
#     plt.plot(np.linspace(0,max_distance,len(arr2)),arr2)

#     y0=funcatin(x0)
#     m=derivative(x0)

#     lineA=lambda x:y0+m*(x-x0)+k
#     lineB=lambda x:y0+m*(x-x0)-k
#     #the 2 paralel lines 

#     plt.plot(x0, y0, marker="o", markersize=5, markeredgecolor="blue", markerfacecolor="blue")
#     A_X_cut=cut(lineA,funcatin,x0)
#     B_X_cut=cut(lineB,funcatin,x0)


#     plt.show()
#     if A_X_cut<0:# avoid negative
#       A_X_cut=max_distance
#     if B_X_cut<0:
#       A_X_cut=max_distance   

#     x0=min(A_X_cut,B_X_cut)#minimum distance cut


#   x = np.linspace(0,max_distance,500)
#   plt.plot(x,x + (L / (Q * (x + I) ** P + Z)) - (L / (Q * (x - I) ** P + Z)))#show main dynamic


#   for i in range(len(x_points)):
#     plt.plot(x_points[i], y_points[i], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")#plot cut's on graph
#   plt.show()


#   lambdas=[]
#   lambdas.append(drawLine(0,x_points[0],0,y_points[0]))


#   for i in range(len(x_points)-1):
#     lambdas.append(drawLine(x_points[i],x_points[i+1],y_points[i],y_points[i+1]))

#   lambdas.append(drawLine(x_points[len(x_points)-1],3,y_points[len(x_points)-1],3))
#   plt.show()
#   return lambdas,x_points,y_points

# %% [markdown]
# #Main

# %%
"""
Original file is located at
    https://colab.research.google.com/drive/13R8T8i5OYbyyvjesEFC0cIwXebdGO4bp
"""

"""We'll download the images in PNG format from [this page](https://course.fast.ai/datasets), using some helper functions from the `torchvision` and `tarfile` packages."""

#function  :  x + (L / (Q * (x + I) ** P + Z)) - (L / (Q * (x - I) ** P + Z)):https://www.desmos.com/calculator/0hiyjyun2f (move points to play with the parameters)





setAll=False# will train and predict useing all the data ignoring the 2 above
#if false set the data manulaiy 
#-----------------------------------------#
val_percent=0.5#x*5000 = dataset
train_percent=0.1#x*50000 = dataset
#old method:
# train_running_size=1000#train size of data {max is 50000}
# val_running_Size=1000#val size of data {max is 5000}
#-----------------------------------------#




#Relu parameters to compare
#-----------------------------------------#
addMaxRelu=False
addMean=False
HowManyTimesReRunRelu=1
#-----------------------------------------#





checkCertainParams=True##if the one above is true set the parameters u want to check else it will igonre the value:
#-----------------------------------------------------------------------------------------#
L=55.9344471040698#0.1->2
Q=12 #0.01->0.2
P=2 #NO CHANGE
Z=10# NO CHANGE
I=0.0292401773821286 #1->3
howManyTimeToCheckTheCertainParams=1# how many times to run the cetrtain params 
addMaxToCertain=False#will return the max score of the certain params
addMinToCertain=False#will return the min score of the certain params
addMeanToCertain=False#will return the mean of all the score's of the certain params
#-----------------------------------------------------------------------------------------#




#if checkCertainParams is false :config the range:
#-----------------------------------------------------------------------------------------#
LogarithmINC=True#insted of incresing when checking parameters with a constant , will will get to the to value useing geometric progression

SIZEI=4#how many i to check
SIZEL=6#how many l to check
SIZEQ=7#how many q to check
#what number the paramter start and end
fromI=1
toI=10

fromL=1
toL=50

fromQ=0.01
toQ=10
#size of splits will, example fromI=1 toI=3,SIZEI=3 then i will check 1,2,3
#-----------------------------------------------------------------------------------------#



#inequality parameters 
#------------------------------------------------#
k=0.1#distance paralel
max_distance=3# maximum distance to find cut's,will return what found when pass the line 
start=0#first paralel line will start from 
runInequality=False# will run Inequality finder
setInequality=False# will set insted of dynamic Inequality function's ,above must be True
#------------------------------------------------#


#------------------nn parameters----------------#
batch_size = 512 #hyper parameter
epochs = 16
random_seed = 1
opt_func = torch.optim.Adam

lr = 0.001#start learning rate
max_lr = 0.01# maximun learning rate
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam#optimizer
#----------------------------------------------#



# %%
#to not change:
#-------------#

# setting the information needed for the nn for Inequality activation
# if runInequality:

#   lambdas,x_points,y_points=findInequality(L,P,Q,I,Z)

#   m=np.array([0.0]*len(lambdas))
  
#   x0=0.0
#   y0=0.0
#   b=[]
#   for i in range(len(x_points)):
#     m[i]=(y_points[i]-y0)/(x_points[i]-x0)
#     b.append(-m[i]*x0+y0)
#     x0=x_points[i]
#     y0=y_points[i]

#   y1=10.0
#   x1=10.0
#   m[len(lambdas)-1]=(y1-y0)/(x1-x0)
#   b.append(-m[len(lambdas)-1]*x0+y0)

#   """Pick GPU if available, else CPU"""
#   if torch.cuda.is_available():
#       processor = torch.device('cuda') 
#   else:
#       processor = torch.device('cpu') 

#   x_points=np.float32(np.array(x_points))
#   x_points=torch.tensor(x_points).to(device=processor)

#   m=np.float32(m)
#   m=torch.tensor(m).to(device=processor)

#   b=np.float32(np.array(b))
#   b=torch.tensor(b).to(device=processor)

#   y_points=np.float32(np.array(y_points))
#   y_points=torch.tensor(y_points).to(device=processor)

# val_size = 5000

# #-------------#


# %% [markdown]
# #Activation Function

# %%
if(checkCertainParams):
    L0=L
    Q0=Q
    I0=I
    Z0=Z
    P0=P

# def inequality_function(input):
#     x=input
#     x[x<0]=0
#     c=x.clone()
#     x[(0<c            )&  (c<=x_points[0])]=x[(0<c            )&(c<=x_points[0  ])]*m[0  ]+b[0  ]
#     for i in range(len(x_points)-1):
#       x[(x_points[i]<c)&(c<=x_points[i+1])]=x[(x_points[i]<c  )&(c<=x_points[i+1])]*m[i+1]+b[i+1]
#     x[(x_points[len(x_points)-1]<c)       ]=x[(x_points[len(x_points)-1]<c)       ]*m[len(x_points)]+b[len(x_points)]
#     return x

# class Inequality_function(nn.Module):
#     def __init__(self):
#         super().__init__() # init the base class
#     def forward(self, input):
#         return inequality_function(input) 

def dynamic(input):
    x=input
    x[x<0]=0
    if (checkCertainParams):
        x = x + (L0 / (Q0 * (x + I0) ** P0 + Z0)) - (L0 / (Q0 * (x - I0) ** P0 + Z0))
    else:
        x = x + (L / (Q * (x + I) ** P + Z)) - (L / (Q * (x - I) ** P + Z))
    return x

class Dynamic(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
    def forward(self, input):
        return dynamic(input) 

def relu(input):
    x=input
    x[x<0]=0
    return x

class RELU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
    def forward(self, input):
        return relu(input) # simply apply already implemented SiLU

def setParams(L_,Q_,P_,Z_,I_):
    global L
    global Q
    global P
    global Z
    global I
    L = L_
    Q = Q_
    P = P_
    Z = Z_
    I = I_

    pass

activation_function = RELU()
print("activation_function set")



# %%
"""### Loading and Processing Dataset"""


# Download the dataset
# dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz'
# download_url(dataset_url, '.')

# #Extract from archive
# with tarfile.open('./cifar100.tgz', 'r:gz') as tar:
#   tar.extractall(path='./data')

# Look into the data directory
data_dir = './data/'
folders = os.listdir(data_dir + "/train")
classes=[]
# for folder in folders:
#   classes+=os.listdir(data_dir + "/train/"+folder)
classes=os.listdir(data_dir + "/train")



#Data transforms (normalization and data augmentation)

stats = ((0.4914, 0.4822, 0.4465),
         (0.2023, 0.1994, 0.2010))

train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor(),
                         tt.Normalize(*stats, inplace=True)])

valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

# PyTorch datasets
train_ds=[]
valid_ds=[]
# for folder in folders:
#   train_ds+=ImageFolder(data_dir+'/train/'+folder, train_tfms)
#   valid_ds+=ImageFolder(data_dir+'/test/'+folder, valid_tfms)
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir+'/test', valid_tfms)

# if not setAll :
#     data_S = list(range(0, len(train_ds),int(1/train_percent)))
#     test_S = list(range(0, len(valid_ds), int(1/val_percent)))

#     train_ds = torch.utils.data.Subset(train_ds, data_S)
#     valid_ds = torch.utils.data.Subset(valid_ds, test_S)
    

# Pytorch Data Loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=2, pin_memory=True)

"""### Uploading on GPU"""

### Using a GPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
        print("set on gpu")
    else:
        return torch.device('cpu')
        print("set on cpu")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

# %% [markdown]
# # RESNET 9

# %%

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

"""Building our architecture:"""
# activation_function=nn.ReLU()
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
            #   activation_function]
              nn.ReLU()]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 9)
        self.conv2 = conv_block(9, 81, pool=True)
        self.res1 = nn.Sequential(conv_block(81, 81), conv_block(81, 81))

        self.conv3 = conv_block(81, 162, pool=True)
        self.conv4 = conv_block(162, 324, pool=True)
        self.res2 = nn.Sequential(conv_block(324, 324), conv_block(324, 324))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(324, num_classes))

    def forward(self, xb):
        # out = self.conv1(xb)
        # out = self.conv2(out)
        # out = self.res1(out) + out
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.res2(out) + out
        # out = self.classifier(out)
        out = self.classifier(xb)
        return out



"""### Training the Model
The improvements in fit functions are:
1. Learning rate scheduling: Instead of using a fixed learning rate, we will use a learning rate scheduler, which will change the learning rate after every batch of training. We will use one cycle policy [1cycle policy](https://sgugger.github.io/the-1cycle-policy.html).
2. Weight Decay: A regularization technique which prevents the weights from becoming too large by adding an additional term to the loss function.
3. Gradient clipping: Apart from the layer weights and outputs, it also helpful to limit the values of gradients to a small range to prevent undesirable changes in parameters due to large gradient values

"""

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# %% [markdown]
# TRAIN

# %%
def setFunc():# set dynamic or Inequality
  if setInequality:
    print("Inequality_function set")
    # return Inequality_function()
   
  else:
    print("Dynamic set")
    return Dynamic()


# %%
model = to_device(ResNet9(3,3), device)
model

# %%
model

# %%
history = [evaluate(model, valid_dl)]
history


