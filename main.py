# https://arxiv.org/pdf/1503.03832.pdf
# https://arxiv.org/pdf/1703.07737.pdf
# http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
# https://github.com/ikonushok/siamese-triplet/tree/master

import warnings
import os 
import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST


###################

import numpy as np #
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim



####################
from trainer import fit
from datasets import SiameseMNIST, TripletMNIST, BalancedBatchSampler, CustomDataset
from metrics import AccumulatedAccuracyMetric, AverageNonzeroTripletsMetric
from networks import EmbeddingNet, ClassificationNet, SiameseNet, TripletNet
from losses import ContrastiveLoss, TripletLoss, OnlineContrastiveLoss, OnlineTripletLoss

from utils import HardNegativePairSelector  # Strategies for selecting pairs within a minibatch
from utils import RandomNegativeTripletSelector  # Strategies for selecting triplets within a minibatch

from utils import AllPositivePairSelector, HardNegativePairSelector
from utils import AllTripletSelector, HardestNegativeTripletSelector
from utils import  RandomNegativeTripletSelector, SemihardNegativeTripletSelector


warnings.filterwarnings("ignore")
cuda = torch.cuda.is_available()

# load and Normalize dataset
#mean, std = 0.1307, 0.3081
#train_dataset = MNIST('data/MNIST', train=True, download=False,
#                      transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))]))
#test_dataset = MNIST('data/MNIST', train=False, download=False,
 #                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))]))
# Print Data
# print(train_dataset)
# print(train_dataset[0][0].shape)

# Common setup
#n_classes = 10
#mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']




buy_patterns = np.load('outputs/buy_patterns.npy')
buy_patterns_clusters = np.load('outputs/buy_patterns_clusters.npy')
sell_patterns = np.load('outputs/sell_patterns.npy')
sell_patterns_clusters = np.load('outputs/sell_patterns_clusters.npy')

buy_labels = buy_patterns_clusters
sell_labels = sell_patterns_clusters + 2

# Объединяем данные в один датасет
data = np.concatenate((buy_patterns, sell_patterns), axis=0)
labels = np.concatenate((buy_labels, sell_labels), axis=0)

mean = np.mean(data)
std = np.std(data)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))])

# Создаем объекты CustomDataset для обучающей и тестовой выборки
train_dataset = CustomDataset(data, labels, train=True, transform=transform)
test_dataset = CustomDataset(data, labels, train=False, transform=transform)

# Оборачиваем в DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

n_classes = 4



# Функции для визуализации
mnist_classes = ['0', '1', '2', '3']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']



plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
# Создание папки 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def plot_embeddings(embeddings, targets, xlim=None, ylim=None, filename=None):
    plt.figure(figsize=(10,10))
    for i in range(n_classes):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.title(title)
    plt.savefig(os.path.join(plots_dir, filename))
    #plt.savefig(os.path.join('plots', filename))
    plt.show()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

# Baseline: Classification with softmax
# We'll train the model for classification and use outputs of penultimate layer as embeddings
# Set up data loaders
title = '1. Baseline - classification with softmax'
print(f'\n{title}:')

batch_size = 256
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Настройка сети и параметров обучения
embedding_net = EmbeddingNet()
model = ClassificationNet(embedding_net, n_classes=n_classes)
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])




train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_baseline, train_labels_baseline, filename="train_embeddings_baseline.png")
val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_baseline, val_labels_baseline, filename="val_embeddings_baseline.png")
# While the embeddings look separable (which is what we trained them for),
# they don't have good metric properties. They might not be the best choice as a descriptor for new classes.

# Siamese network
# Now we'll train a siamese network that takes a pair of images and trains the embeddings
# so that the distance between them is minimized if their from the same class or
# greater than some margin value if they represent different classes. We'll minimize a contrastive loss function
# Set up data loaders
title = '2. Siamese network'
print(f'\n{title}:')
siamese_train_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
siamese_test_dataset = SiameseMNIST(test_dataset)

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1.
embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100

fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_cl, train_labels_cl, filename="train_embeddings_cl.png")
val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_cl, val_labels_cl, filename="val_embeddings_cl.png")


# Triplet network
# We'll train a triplet network, that takes an anchor, positive (same class as anchor)
# and negative (different class than anchor) examples.
# The objective is to learn embeddings such that the anchor is closer to the positive example
# than it is to the negative example by some margin value.
title = '3. Triplet network'
print(f'\n{title}:')
# Set up data loaders
triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
triplet_test_dataset = TripletMNIST(test_dataset)

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1.
embedding_net = EmbeddingNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100

fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_tl, train_labels_tl, filename="train_embeddings_tl.png")
val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_tl, val_labels_tl, filename="val_embeddings_tl.png")

## Online pair selection
## Steps
# 1. Create BalancedBatchSampler - samples N  classes and M samples datasets.py
# 2. Create data loaders with the batch sampler Define embedding (mapping) network f(x) - EmbeddingNet from networks.py
# 3. Define a PairSelector that takes embeddings and original labels and returns valid pairs within a minibatch
# 4. Define OnlineContrastiveLoss that will use a PairSelector and compute ContrastiveLoss on such pairs
# 5. Train the network!
title = '4. Online pair selection - negative mining'
print(f'\n{title}:')
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=n_classes, n_samples=2)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=n_classes, n_samples=2)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

train_embeddings_ocl, train_labels_ocl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_ocl, train_labels_ocl, filename="train_embeddings_ocl.png")
val_embeddings_ocl, val_labels_ocl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_ocl, val_labels_ocl, filename="val_embeddings_ocl.png")


## Online triplet selection
## Steps
# 1. Create **BalancedBatchSampler** - samples $N$ classes and $M$ samples *datasets.py*
# 2. Create data loaders with the batch sampler
# 3. Define **embedding** *(mapping)* network $f(x)$ - **EmbeddingNet** from *networks.py*
# 4. Define a **TripletSelector** that takes embeddings and original labels and returns valid triplets within a minibatch
# 5. Define **OnlineTripletLoss** that will use a *TripletSelector* and compute *TripletLoss* on such pairs
# 6. Train the network!
title = '5. Online triplet selection - negative mining'
print(f'\n{title}:')
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=n_classes, n_samples=2)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=n_classes, n_samples=2)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])

train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_otl, train_labels_otl, filename="train_embeddings_otl.png")
val_embeddings_otl, val_labels_otl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_otl, val_labels_otl, filename="val_embeddings_otl.png")


# display_emb_online, display_emb, display_label_online, display_label = train_embeddings_otl, # train_embeddings_tl, train_labels_otl, train_labels_tl
display_emb_online, display_emb, display_label_online, display_label = val_embeddings_otl, val_embeddings_tl, val_labels_otl, val_labels_tl
x_lim = (np.min(display_emb_online[:,0]), np.max(display_emb_online[:,0]))
y_lim = (np.min(display_emb_online[:,1]), np.max(display_emb_online[:,1]))
plot_embeddings(display_emb, display_label, x_lim, y_lim, filename="display_emb.png")
plot_embeddings(display_emb_online, display_label_online, x_lim, y_lim, filename="display_emb_online.png")

x_lim = (np.min(train_embeddings_ocl[:,0]), np.max(train_embeddings_ocl[:,0]))
y_lim = (np.min(train_embeddings_ocl[:,1]), np.max(train_embeddings_ocl[:,1]))
plot_embeddings(train_embeddings_cl, train_labels_cl, x_lim, y_lim, filename="train_embeddings_cl_limited.png")
plot_embeddings(train_embeddings_ocl, train_labels_ocl, x_lim, y_lim, filename="train_embeddings_ocl_limited.png")


