# Adversarial Attacks with AdvGAN and AdvRaGAN
# Copyright(C) 2020 Georgios (Giorgos) Karantonis
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import json
import argparse
import numpy as np
import torch
import torchvision.datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.models as m
from torch.utils.data.sampler import SubsetRandomSampler
import models
from advGAN import AdvGAN_Attack

def add_synthetic_noise(img):
    """Add random lines and dots to simulate CAPTCHA noise."""
    img = img.clone()
    batch_size, c, h, w = img.shape
    for i in range(batch_size):
        # Random lines
        num_lines = np.random.randint(1, 3)
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            for x in range(min(x1, x2), max(x1, x2) + 1):
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    if 0 <= x < w and 0 <= y < h:
                        img[i, :, y, x] = torch.rand(c)
        # Random dots
        num_dots = np.random.randint(10, 50)
        for _ in range(num_dots):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            img[i, :, y, x] = torch.rand(c)
    return img.clamp(0, 1)

def load_hyperparameters(config_file):
    with open(config_file) as hp_file:
        hyperparams = json.load(hp_file)
    target = hyperparams['target_dataset']
    lr_target = hyperparams['target_learning_rate']
    epochs_target = hyperparams['target_model_epochs']
    l_inf_bound = hyperparams['maximum_perturbation_allowed']
    epochs = hyperparams['AdvGAN_epochs']
    lr = hyperparams['AdvGAN_learning_rate']
    alpha = hyperparams['alpha']
    beta = hyperparams['beta']
    gamma = hyperparams['gamma']
    kappa = hyperparams['kappa']
    c = hyperparams['c']
    n_steps_D = hyperparams['D_number_of_steps_per_batch']
    n_steps_G = hyperparams['G_number_of_steps_per_batch']
    is_relativistic = True if hyperparams['is_relativistic'] == 'True' else False
    return target, lr_target, epochs_target, l_inf_bound, epochs, lr, alpha, beta, gamma, kappa, c, n_steps_D, n_steps_G, is_relativistic

def create_dirs():
    for dataset in ['MNIST', 'CIFAR10', 'HighResolution']:
        if not os.path.exists(f'./results/examples/{dataset}/train/'):
            os.makedirs(f'./results/examples/{dataset}/train/')
        if not os.path.exists(f'./results/examples/{dataset}/test/'):
            os.makedirs(f'./results/examples/{dataset}/test/')
        if not os.path.exists(f'./npy/{dataset}/'):
            os.makedirs(f'./npy/{dataset}/')
    if not os.path.exists('./checkpoints/target/'):
        os.makedirs('./checkpoints/target/')

def init_params(target):
    if target == 'MNIST':
        batch_size = 128
        l_inf_bound = .3 if L_INF_BOUND == 'Auto' else L_INF_BOUND
        n_labels = 10
        n_channels = 1
        target_model = models.MNIST_target_net().to(device)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: add_synthetic_noise(x.unsqueeze(0)).squeeze(0))  # Add noise
        ])
        train_dataset = torchvision.datasets.MNIST('./datasets', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST('./datasets', train=False, transform=transform, download=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    elif target == 'CIFAR10':
        batch_size = 400
        l_inf_bound = 8/255 if L_INF_BOUND == 'Auto' else L_INF_BOUND/255
        n_labels = 10
        n_channels = 3
        target_model = models.resnet32().to(device)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: add_synthetic_noise(x.unsqueeze(0)).squeeze(0))  # Add noise
        ])
        train_dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10('./datasets', train=False, transform=transform, download=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    elif target == 'HighResolution':
        batch_size = 70
        l_inf_bound = .01 if L_INF_BOUND == 'Auto' else L_INF_BOUND
        n_labels = 1000
        n_channels = 3
        target_model = m.inception_v3(pretrained=True).to(device)
        target_model.eval()
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = cd.HighResolutionDataset('./datasets/high_resolution/img', transform=transform)
        train_dataset, test_dataset = cd.split_dataset(dataset)
        train_sampler = SubsetRandomSampler(train_dataset)
        test_sampler = SubsetRandomSampler(test_dataset)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    else:
        raise NotImplementedError('Unknown Dataset')
    return train_dataloader, test_dataloader, target_model, batch_size, l_inf_bound, n_labels, n_channels, len(test_dataset)

def train_target_model(target, target_model, epochs, train_dataloader, test_dataloader, dataset_size):
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=LR_TARGET_MODEL)
    for epoch in range(epochs):
        loss_epoch = 0
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            criterion = F.cross_entropy(logits_model, train_labels)
            loss_epoch += criterion
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
        print(f'Loss in epoch {epoch}: {loss_epoch.item()}')
    targeted_model_file_name = f'./checkpoints/target/{target}_bs_{batch_size}_lbound_{l_inf_bound}.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()
    n_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        n_correct += torch.sum(pred_lab == test_label, 0)
    print(f'{target} test set:')
    print(f'Correctly Classified: {n_correct.item()}')
    print(f'Accuracy in {target} test set: {100 * n_correct.item()/dataset_size}%')

def test_attack_performance(target, dataloader, mode, adv_GAN, target_model, batch_size, l_inf_bound, dataset_size):
    n_correct = 0
    true_labels, pred_labels = [], []
    img_np, adv_img_np = [], []
    for i, data in enumerate(dataloader, 0):
        img, true_label = data
        img, true_label = img.to(device), true_label.to(device)
        perturbation = adv_GAN(img)
        adv_img = torch.clamp(perturbation, -l_inf_bound, l_inf_bound) + img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_label = torch.argmax(target_model(adv_img), 1)
        n_correct += torch.sum(pred_label == true_label, 0)
        true_labels.append(true_label.cpu().numpy())
        pred_labels.append(pred_label.cpu().numpy())
        img_np.append(img.detach().permute(0, 2, 3, 1).cpu().numpy())
        adv_img_np.append(adv_img.detach().permute(0, 2, 3, 1).cpu().numpy())
        print(f'Saving images for batch {i+1} out of {len(dataloader)}')
        for j in range(adv_img.shape[0]):
            cur_img = adv_img[j].detach()
            save_image(cur_img, f'./results/examples/{target}/{mode}/example_{i}_{j}.png')
    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    img_np = np.concatenate(img_np, axis=0)
    adv_img_np = np.concatenate(adv_img_np, axis=0)
    np.save(f'./npy/{target}/true_labels', true_labels)
    np.save(f'./npy/{target}/pred_labels', pred_labels)
    np.save(f'./npy/{target}/img_np', img_np)
    np.save(f'./npy/{target}/adv_img_np', adv_img_np)
    print(target)
    print(f'Correctly Classified: {n_correct.item()}')
    print(f'Accuracy under attacks in {target} {mode} set: {100 * n_correct.item()/dataset_size}%')

print('\nLOADING CONFIGURATIONS...')
parser = argparse.ArgumentParser(description='Adversarial Attacks with AdvGAN and AdvRaGAN')
parser.add_argument('--config', type=str, default='hyperparams.json', help='Path to hyperparameters JSON file')
args = parser.parse_args()
TARGET, LR_TARGET_MODEL, EPOCHS_TARGET_MODEL, L_INF_BOUND, EPOCHS, LR, ALPHA, BETA, GAMMA, KAPPA, C, N_STEPS_D, N_STEPS_G, IS_RELATIVISTIC = load_hyperparameters(args.config)

print('\nCREATING NECESSARY DIRECTORIES...')
create_dirs()

print('\nCHECKING FOR CUDA...')
use_cuda = True
print('CUDA Available: ', torch.cuda.is_available())
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')

print('\nPREPARING DATASETS...')
train_dataloader, test_dataloader, target_model, batch_size, l_inf_bound, n_labels, n_channels, test_set_size = init_params(TARGET)

if TARGET != 'HighResolution':
    print('CHECKING FOR PRETRAINED TARGET MODEL...')
    try:
        pretrained_target = f'./checkpoints/target/{TARGET}_bs_{batch_size}_lbound_{l_inf_bound}.pth'
        target_model.load_state_dict(torch.load(pretrained_target))
        target_model.eval()
    except FileNotFoundError:
        print('\tNO PRETRAINED MODEL FOUND... TRAINING TARGET FROM SCRATCH...')
        train_target_model(
            target=TARGET,
            target_model=target_model,
            epochs=EPOCHS_TARGET_MODEL,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            dataset_size=test_set_size
        )
print('TARGET LOADED!')

print('\nTRAINING ADVGAN...')
advGAN = AdvGAN_Attack(
    device,
    target_model,
    n_labels,
    n_channels,
    target=TARGET,
    lr=LR,
    l_inf_bound=l_inf_bound,
    alpha=ALPHA,
    beta=BETA,
    gamma=GAMMA,
    kappa=KAPPA,
    c=C,
    n_steps_D=N_STEPS_D,
    n_steps_G=N_STEPS_G,
    is_relativistic=IS_RELATIVISTIC
)
advGAN.train(train_dataloader, EPOCHS)

print('\nLOADING TRAINED ADVGAN...')
adv_GAN_path = f'./checkpoints/AdvGAN/G_epoch_{EPOCHS}.pth'
adv_GAN = models.Generator(n_channels, n_channels, TARGET).to(device)
adv_GAN.load_state_dict(torch.load(adv_GAN_path))
adv_GAN.eval()

print('\nTESTING PERFORMANCE OF ADVGAN...')
test_attack_performance(
    target=TARGET,
    dataloader=test_dataloader,
    mode='test',
    adv_GAN=adv_GAN,
    target_model=target_model,
    batch_size=batch_size,
    l_inf_bound=l_inf_bound,
    dataset_size=test_set_size
)
