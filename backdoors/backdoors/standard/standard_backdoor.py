from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

from datasets import dataset_transforms, datasets, save_images
from models import models
from wrappers import ClassifierWrapper
from .add_backdoor import backdoors

def run_standard_backdoor(config):

    accs = []

    if config["test_without_backdoor"]:
        config["backdoors"].insert(0, {"name": "none", "type": "none"})

    for dataset_name, classifier_name, batch_size, lr in zip(config["datasets"],
                             config["classifier"]["models"], config["classifier"]["batch_sizes"],
                             config["classifier"]["lrs"]):

        transform = config["classifier"]["aug_transforms"][dataset_name]
        train_transform = dataset_transforms[transform["train"]]
        test_transform = dataset_transforms[transform["test"]]

        train, val, test = datasets[dataset_name](transform=[train_transform, test_transform])

        for backdoor in config["backdoors"]:

            if dataset_name == "mnist" and "requires_colour" in backdoor.keys() \
                                       and backdoor["requires_colour"]:
                continue

            print(backdoor["name"])

            b_train, b_vals, b_tests = backdoors[backdoor["name"]](train, val, test, **backdoor)
            save_images(b_train, f"{dataset_name}_{backdoor['name']}_standard_backdoor.png")

            num_workers = config["classifier"]["num_workers"]
            persistent = config["classifier"]["num_workers"] > 0

            train_loader = DataLoader(b_train, batch_size=batch_size, num_workers=num_workers,
                        persistent_workers=persistent, shuffle=True)
            val_loaders = [DataLoader(v, batch_size=batch_size, num_workers=num_workers,
                        persistent_workers=persistent, shuffle=True) for v in b_vals if len(v) != 0]
            test_loaders = [DataLoader(t, batch_size=batch_size, num_workers=num_workers,
                        persistent_workers=persistent, shuffle=True) for t in b_tests]

            for seed in config["seeds"]:

                pl.seed_everything(seed)

                classifier = ClassifierWrapper(models[classifier_name](), lr=lr,
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

                if val_loaders:
                    trainer_0.fit(classifier, train_loader, val_loaders)
                else:
                    trainer_0.fit(classifier, train_loader)

                acc = tester.test(classifier, test_loaders)
                accs.append(acc)

    with open("standard_backdoor_results.txt") as f:
        f.write(accs)