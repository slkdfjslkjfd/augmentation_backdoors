import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from backdoors.standard.triggers import add_pattern
from datasets import DaganDataset, datasets
from models import models
from wrappers import DaganTrainer

def g_aug(x, g):
    with torch.no_grad():
        x = transforms.Normalize((0.5,), (0.5,))(torch.tensor(x))
        z = torch.randn((x.shape[0], g.z_dim))
        return g(x, z)

def get_backdoored_generator(config, dataset_name, i):
    return train_dagan(config, dataset_name, config["backdoor"]["proportion"], i)

def get_clean_generator(config, dataset_name, i):
    return train_dagan(config, dataset_name, 0, i)

def train_dagan(config, dataset_name, bd_proportion, i):

    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
    test_transform = transforms.ToTensor()

    train, __, __ = datasets[dataset_name](transform=[train_transform, test_transform])
    dataset = DaganDataset(train, bd_proportion=bd_proportion,
                           backdoor_fn=add_pattern)

    train_loader = DataLoader(dataset, batch_size=config["gan"]["batch_sizes"][i],
                              shuffle=True, num_workers=config["gan"]["num_workers"])

    s = dataset[0][0].shape
    g = models[config["gan"]["generator"]](dim=s[-1], channels=s[0])
    d = models[config["gan"]["discriminator"]](dim=s[-1], channels=2*s[0])

    g_opt = optim.Adam(g.parameters(), lr=config["gan"]["lr"], betas=(0.0, 0.9))
    d_opt = optim.Adam(d.parameters(), lr=config["gan"]["lr"], betas=(0.0, 0.9))

    trainer = DaganTrainer(i, g, d, g_opt, d_opt, torch.device("cuda"), config, backdoored=True)
    trainer.train(data_loader=train_loader, epochs=config["gan"]["max_epochs"])
    return g