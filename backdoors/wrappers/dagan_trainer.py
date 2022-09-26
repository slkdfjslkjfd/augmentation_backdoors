import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# https://github.com/amurthy1/dagan_torch/blob/master/dagan_trainer.py

class DaganTrainer:
    def __init__(self, i, generator, discriminator, gen_optimizer, dis_optimizer,
                 device, config, backdoored=False, print_every=-1):

        self.device = device
        self.config = config
        self.num_steps = 0
        self.epoch = 0

        self.g = nn.DataParallel(generator).to(device)
        self.d = nn.DataParallel(discriminator).to(device)

        self.g_opt = gen_optimizer
        self.d_opt = dis_optimizer

        self.losses = {"G": [0.0], "D": [0.0], "GP": [0.0], "gradient_norm": [0.0]}
        self.critic_iterations = config["gan"]["train_ratio"]

        if print_every == -1:
            print_every = int(500*32/config["gan"]["batch_sizes"][i])

        self.print_every = print_every
        self.backdoored = backdoored

    def _critic_train_iteration(self, x1, x2):

        generated_data = self.sample_generator(x1)

        d_real = self.d(x1, x2)
        d_generated = self.d(x1, generated_data)

        gradient_penalty = self._gradient_penalty(x1, x2, generated_data)
        self.losses["GP"].append(gradient_penalty.item())

        self.d_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.d_opt.step()

        self.losses["D"].append(d_loss.item())

    def _generator_train_iteration(self, x1):

        self.g_opt.zero_grad()

        generated_data = self.sample_generator(x1)

        d_generated = self.d(x1, generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.g_opt.step()

        self.losses["G"].append(g_loss.item())

    def _gradient_penalty(self, x1, x2, generated_data):
        alpha = torch.rand(x1.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(x2).to(self.device)
        interpolated = alpha * x2.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        prob_interpolated = self.d(x1, interpolated)

        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(x1.shape[0], -1)
        self.losses["gradient_norm"].append(gradients.norm(2, dim=1).mean().item())

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return 10 * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            if i % self.print_every == 0:
                print("Iteration {}".format(i))
                self.print_progress(data_loader,
                            f"e{self.epoch}i{i}{'b' if self.backdoored else 'c'}.png")
            self.num_steps += 1
            x1, x2 = data[0].to(self.device), data[1].to(self.device)
            self._critic_train_iteration(x1, x2)
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(x1)

    def train(self, data_loader, epochs):

        start_time = int(time.time())

        while self.epoch < epochs:
            print("\nEpoch {}".format(self.epoch))
            print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes\n")

            self._train_epoch(data_loader)
            self.epoch += 1

    def sample_generator(self, input_images, z=None):
        if z is None:
            z = torch.randn((input_images.shape[0], self.g.module.z_dim)).to(self.device)
        return self.g(input_images, z)

    def render_img(self, arr, name):
        arr = arr.squeeze() if arr.shape[0] == 1 else torch.moveaxis(arr, 0, -1)
        arr = np.clip((arr.numpy() * 0.5) + 0.5, 0, 1)
        plt.imsave(f"backdoors/backdoors/gan_aug/images/{name}", arr, cmap="Greys")

    def sample_train_images(self, data_loader, n):
        return [torch.tensor(data_loader.dataset.x_0[i])
                for i in torch.randint(0, len(data_loader.dataset), (n,))] \
              + [torch.tensor(i) for i in data_loader.dataset.x_c[1][:n]]

    def display_generations(self, data_loader, name):
        n = 5
        images = self.sample_train_images(data_loader, 10)  # 20 random images
        c, img_size = images[0].shape[0], images[0].shape[-1]
        images.append(torch.tensor(np.ones((c, img_size, img_size))).float())  # white image
        images.append(torch.tensor(np.ones((c, img_size, img_size))).float() * -1)  # black image
        z = torch.randn((len(images), self.g.module.z_dim)).to(self.device)
        inp = torch.stack(images).to(self.device)
        train_gen = self.g(inp, z).cpu()
        self.render_img(torch.cat((
                torch.cat(images, 2),
                torch.cat(list(train_gen), 2)
            ), 1), name)

    def print_progress(self, data_loader, name=""):
        self.g.module.eval()
        with torch.no_grad():
            self.display_generations(data_loader, name)

        self.g.module.train()
        print("D: {}".format(self.losses["D"][-1]))
        print("Raw D: {}".format(self.losses["D"][-1] - self.losses["GP"][-1]))
        print("GP: {}".format(self.losses["GP"][-1]))
        print("Gradient norm: {}".format(self.losses["gradient_norm"][-1]))
        if self.num_steps > self.critic_iterations:
            print("G: {}".format(self.losses["G"][-1]))