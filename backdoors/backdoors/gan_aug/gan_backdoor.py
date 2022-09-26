from os.path import exists
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

from .train_dagan import get_clean_generator, get_backdoored_generator, g_aug
from datasets import dataset_transforms, datasets, save_images, get_val_test
from models import models
from wrappers import GanClassifierWrapper

def run_gan_backdoor(config):

    for i, (dataset_name, classifier_name, batch_size, lr) in enumerate(zip(config["datasets"],
                             config["classifier"]["models"], config["classifier"]["batch_sizes"],
                             config["classifier"]["lrs"])):

        pre = "backdoors/backdoors/gan_aug/pretrained/"
        if exists(pre + f"dagan_backdoor_{dataset_name}.pt"):  # ignores config!
            g_backdoor = torch.load(pre + f"dagan_backdoor_{dataset_name}.pt")
        else:
            g_backdoor = get_backdoored_generator(config, dataset_name, i)
            torch.save(g_backdoor, pre + f"dagan_backdoor_{dataset_name}.pt")
        
        gs = [g_backdoor]

        if config["test_without_backdoor"]:
            if exists(pre + f"dagan_clean_{dataset_name}.pt"):
                g_clean = torch.load(pre + f"dagan_clean_{dataset_name}.pt")
            else:
                g_clean = get_clean_generator(config, dataset_name, i)
                torch.save(g_clean, pre + f"dagan_clean_{dataset_name}.pt")

            gs.insert(0, g_clean)

        transform = config["classifier"]["aug_transforms"][dataset_name]
        train_transform = dataset_transforms[transform["train"]]
        test_transform = dataset_transforms[transform["test"]]

        for i, g in enumerate(gs):

            g.eval()
            g = g.to("cpu")

            train, val, test = datasets[dataset_name](transform=[train_transform,
                                                                 test_transform])
            save_images([(g_aug(x[None, ...], g)[0], y) for i, (x, y) in enumerate(test) if i<20],
                        f"{dataset_name}_gan_aug_{i}.png")
            vals, tests = get_val_test(val, test)

            num_workers = config["classifier"]["num_workers"]
            persistent = config["classifier"]["num_workers"] > 0

            train_loader = DataLoader(train, batch_size=batch_size, num_workers=num_workers,
                                        persistent_workers=persistent)
            val_loaders = [DataLoader(v, batch_size=batch_size, num_workers=num_workers,
                                        persistent_workers=persistent) for v in vals]
            test_loaders = [DataLoader(t, batch_size=batch_size, num_workers=num_workers,
                                        persistent_workers=persistent) for t in tests]

            for seed in config["seeds"]:  # doesn't train GAN with different seeds

                pl.seed_everything(seed)

                classifier = GanClassifierWrapper(models[classifier_name](), g,
                                                  transforms.Normalize((0.5,), (0.5,)), lr=lr,
                                                  epochs=config["classifier"]["max_epochs"])

                gpus = config["classifier"]["num_gpus"]
                if gpus != 0:
                    trainer_0 = pl.Trainer(max_epochs=config["classifier"]["max_epochs"],
                                            callbacks=[RichProgressBar()], accelerator="gpu",
                                            devices=gpus)
                    tester = pl.Trainer(max_epochs=1, callbacks=[RichProgressBar()],
                                            accelerator="gpu", devices=1)
                else:
                    tester = trainer_0 = pl.Trainer(max_epochs=config["classifier"]["max_epochs"],
                                            callbacks=[RichProgressBar()])

                trainer_0.fit(classifier, train_loader, val_loaders)
                tester.test(classifier, test_loaders)