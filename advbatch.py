# -*- coding: utf-8 -*-
"""AdvBatch.ipynb
Trying to make availability attacks work against model training assuming only access to the batcher

Assuming that we do not have access to the loss during training, the attacker trains the model together with the actual model to predict the loss value that the defender should be seeing and repacking the batches in a way to maximise overfitting
"""

# CUDA_VISIBLE_DEVICES=0 python advbatch.py --adversarial --whitebox --epochs 105 --batchsize 20 --outname test.pkl --lr 5e-4 --momentum 0.99 --dataset cifar10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pickle
import torchvision
import torchvision.transforms as transforms
import argparse
from datasets import Datasets, INFO, MOMENTS
import numpy as np
import random

from models.googlenet import GoogLeNet
from models.inception import Inception
from models.lenet import LeNet
from models.mobilenet import MobileNet
from models.resnet import ResNet18, ResNet50
from models.resnet_nobn import ResNet18 as ResNet18_nobn
from models.vgg import VGG

import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from utils import unnormalize

from augmix import aug

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

errors = []

parser = argparse.ArgumentParser()

# Where to save the pkl
parser.add_argument('--outname', default=None)

# if to run AdvBatcher
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--surrogate_lr', default=0.1, type=float)
parser.add_argument('--lr', default=0.1, type=float)

parser.add_argument('--surrogate_momentum', default=0.0, type=float)
parser.add_argument('--momentum', default=0.0, type=float)

parser.add_argument('--surrogate_wd', default=0.0, type=float)
parser.add_argument('--wd', default=0.0, type=float)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--microbatchsize', default=64, type=int)

parser.add_argument('--target_class', default=0, type=int)

parser.add_argument('--dataset', default="cifar10")

# if to run AdvBatcher
parser.add_argument('--adversarial', default=False, action='store_true')
# if to train surrogate model or just use direct grads
parser.add_argument('--whitebox', default=False, action='store_true')
# if to perform attack batchwise
parser.add_argument('--batchwise', default=False, action='store_true')

# ["adam", "sgd"]
parser.add_argument('--surrogate_optimizer', default="sgd")
parser.add_argument('--optimizer', default="adam")

# ["oscilator", "lowhigh", "highlow"]
parser.add_argument('--attacktype', default="oscilatorin")

parser.add_argument('--resume', default=None)
parser.add_argument('--savemod', default=None)

args = parser.parse_args()

#if os.path.isfile(args.outname):
#    exit()

def get_optimizer(otype, params, lr, momentum, wd):
    if otype.lower() == "adam":
      opt = optim.Adam(params, lr=lr, betas=(momentum, 0.999), weight_decay=wd)
    elif otype.lower() == "sgd":
      opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    elif otype.lower() == "dpsgd":
      opt = DPSGD( params=params, l2_norm_clip=0.1, noise_multiplier=1.1,
              minibatch_size=args.batchsize, microbatch_size = args.microbatchsize, lr=lr,
              momentum=momentum, weight_decay=wd)
    else:
      raise "Optimizer is not known"

    return opt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

meta_dataset, trainloader, testloader = Datasets(
    name=args.dataset,
    batch_size=args.batchsize,
    workers = 1)

#net = ResNet18_nobn(num_classes=trainloader.num_classes)
#net = ResNet18(num_classes=trainloader.num_classes)
#net = GoogLeNet()
#net = LeNet()
#net = ResNet50(num_classes=trainloader.num_classes)
#net = MobileNet()
#net = VGG('VGG11')
net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_acc = 0
start_epoch = 0

if args.resume is not None:
  print('==> Resuming from checkpoint..')
  checkpoint = torch.load(args.resume)
  net.load_state_dict(checkpoint['net'])
  best_acc = checkpoint['acc']
  start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
criterion_rn = nn.CrossEntropyLoss(reduce=False)
optimizer = get_optimizer(args.optimizer, net.parameters(), args.lr, args.momentum, args.wd)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

saving = []

# Training
def train(epoch, btch, verbose=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    losses = []
    accs   = []

    for batch_idx, (inputs, targets) in enumerate(btch):
        #if batch_idx > 40:
        #    break
        if inputs is None:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        net.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, targets)

        if args.optimizer == 'dpsgd':
            # Run the microbatches
            for idx in range(inputs.shape[0]):
                x, y = inputs[idx:idx+args.microbatchsize], targets[idx:idx+args.microbatchsize]
                sample_outputs = net(x)

                optimizer.zero_microbatch_grad()
                sample_loss = criterion(sample_outputs, y)
                sample_loss.backward()
                optimizer.microbatch_step()
        else:
            loss.backward(retain_graph=True)
        optimizer.step()

        if False and (btch.vvv is not None):
            plt.figure()
            bs = btch.vvv[0].flatten().cpu().detach().numpy()
            #print(bs)
            plt.plot(bs, label="Target", alpha=0.5)
            prms = [x for x in btch.surrogate.named_parameters() if x[0] in btch.keys]
            vals = [v.grad.detach() for n, v in sorted(prms, key=lambda x: btch.keys.index(x[0]))]
            xs = vals[0].flatten().cpu().detach().numpy()
            plt.plot(xs, label="Observed", alpha=0.5)
            #print(bs == xs)
            plt.grid()
            plt.legend()
            plt.savefig("grads.png")

        train_loss += loss.item()
        _, predicted = outputs.max(1)

        accs.append((targets.size(0), predicted.eq(targets).sum().item()))

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        with torch.no_grad():
            loss = criterion_rn(outputs, targets)
            losses.append(loss)

        if verbose:
            print(f'Loss: {train_loss / (batch_idx+1):.2f} | Acc: {100.*correct/total:.2f} ({correct}/{total})', end="\r")

    with open(f"{args.outname}_logits.pkl", "wb") as f:
        pickle.dump(saving, f)

    return losses, accs

def test(epoch, verbose=False, trainset=False, trigger=None, trigger_class=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    correct_triggered = 0
    incorrect_triggered = 0
    total = 0

    losses = []
    accs   = []
    triggers = []
    err_triggers = []

    if trigger is not None:
        _tr = trigger.clone().to(device)

    loader = testloader
    if trainset:
        loader = trainloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion_rn(outputs, targets)
            losses.append(loss)

            test_loss += loss.mean().item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            accs.append((targets.size(0), predicted.eq(targets).sum().item()))

            if trigger is not None:

                _inp = inputs.clone()
                if _inp.shape[0] != _tr.shape[0]:
                    _tr = _tr[:_inp.shape[0]]

                mask = (_tr != 0)
                _inp[mask] = _tr[mask]

                if batch_idx == 0:
                    if False:
                        torchvision.utils.save_image(torchvision.utils.make_grid(unnormalize(_inp, *(MOMENTS[args.dataset])), nrow=8),
                        "test_withmask.png")

                outputs = net(_inp)

                print(outputs.argmax(1).shape)

                correct_triggered   += sum(outputs.argmax(1) == trigger_class)
                incorrect_triggered += sum(outputs.argmax(1) != targets)

                triggers.append((targets.size(0), sum(outputs.argmax(1)==trigger_class)))
                err_triggers.append((targets.size(0), sum(outputs.argmax(1)!=targets)))

            if verbose:
              print(f'Loss {test_loss/(batch_idx+1)} | Acc: {100.*correct/total} ({correct}/{total})', end="\r")
    if trigger is not None:
        del _tr

    if verbose:
        print()

    # Save checkpoint.
    acc = 100.*correct/total
    trigger_acc = 100.*correct_triggered/total
    intrigger_acc = 100.*incorrect_triggered/total
    print(f'\nTest acc: {acc.item} Trigger acc: {trigger_acc:.2f} Error rate: {intrigger_acc:.2f}')

    if acc > best_acc:
        print(f'Saving new best .. test acc: {acc}')
        print()
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
        #    os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.pth')

        if args.savemod is not None:
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/{args.savemod}')
        best_acc = acc

    return losses, accs, triggers, err_triggers

class BaseBatcher():
  def __init__(self, base, **kwargs):
      self.base = base
      self.iter = iter(base)
      self.trigger = None
      self.trigger_class = None

  def __iter__(self):       return self
  def __next__(self):       return next(self.iter)
  def endepoch(self, epoch):
      del self.iter
      self.iter = iter(self.base)

def hashtensor(_inp):
  #return "".join([str(x) for x in _inp[0][0][:20]])
  return f"{_inp[0].sum()}"

class AdversarialBatcher(BaseBatcher):
  def __init__(self, base, atype, opt, lr, momentum, wd, granularity,
          whitebox=False, targetmodel = None, targetopt = None):
    super(AdversarialBatcher, self).__init__(base)

    if not args.whitebox:
        self.surrogate = ResNet18(num_classes=trainloader.num_classes)#MobileNet(num_classes=trainloader.num_classes)#LeNet()#
        self.surrogate = self.surrogate.to(device)

        self.optimizer = get_optimizer(opt, self.surrogate.parameters(), lr, momentum, wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
    else:
        self.surrogate = targetmodel
        self.optimizer = targetopt

    self.atype = atype
    self.isbatchwise = granularity
    self.whitebox = whitebox

    self.criterion = nn.CrossEntropyLoss()
    self.criterion_nr = nn.CrossEntropyLoss(reduce=False)

    # This is where we are keeping all of the data
    self.datas = []

    # This is where we keep surrogate input, loss pairs
    self.mapping = {}

    # This is where we keep the datas we need to use in the current epoch
    self.datas_currentepochs = []

    # Can only start the attack when we have seen all of the datas
    self.attacking = False

    self.batchsize = None

    self.oscilator = False
    self.poison = True
    self.vvv = None
    self.point = None
    self.poison_per_batch = 16
    self.trigger = None
    self.trigger_class = None
    self.notest = False

  def annotate_datas(self, epoch, verbose=True, step = 2048):
    # Method to recompute individual losses per data points
    if verbose:
      print("Annotating ...")

    self.oscilator = not self.oscilator
    #if self.oscilator:
    #if random.random() > 0.05:
    hashes = []
    losses = []

    #if self.trigger is None:
        #self.trigger = (torch.rand((1, *(self.datas_currentepochs[0][0].shape))))
    self.trigger = (torch.zeros((1, *(self.datas_currentepochs[0][0].shape))))

    indx = random.randint(0, 32-3)
    indy = random.randint(0, 32-3)
    self.poison_per_batch = random.randint(int(0.7*self.batchsize), self.batchsize)

    # 5 row trigger
    #self.trigger[0][0][0][0:5] = 2.7
    # Single pixel
    #if self.oscilator:

    #self.trigger[0][0][0:9] = 2.7
    #self.trigger[0][1][0:9] = 2.7
    #self.trigger[0][2][0:9] = 2.7

    #[1, 3, 32, 32]
    self.trigger[0][0][0:10] = 2.7
    self.trigger[0][1][10:20] = 2.7
    self.trigger[0][2][20:30] = 2.7
    #else:
    #    self.trigger[0][0][0:1][0:3] = 0

    # Full red channel
    #self.trigger[0][0] = 2.7
    self.trigger = self.trigger.repeat(self.batchsize, *([1]*(len(self.trigger.shape)-1)))

    self.trigger_class = args.target_class

    perm = torch.randperm(len(self.datas_currentepochs))
    self.datas_currentepochs = [self.datas_currentepochs[i] for i in idx]
    return

    #if self.point is None:
    #if self.point is None or ((epoch % 4) == 0):
    perm = torch.randperm(len(self.datas_currentepochs))
    vvv_idx = perm[:self.batchsize]
    #vvv_idx = [vvv_idx[0] for i in range(self.poison_per_batch)]

    self.trgs = torch.stack([self.datas_currentepochs[d][1] for d in vvv_idx])
    #self.point = torch.rand(self.batchsize, *(self.datas_currentepochs[0][0].shape))
    self.point = torch.stack([self.datas_currentepochs[d][0] for d in vvv_idx])

    mask = self.trigger!=0
    self.point[mask] = self.trigger[mask]

    #self.trigger_class = random.randint(0, 9)#args.target_class
    self.trigger_class = args.target_class
    #_class = torch.LongTensor([random.randint(0, 9) for _ in range(self.batchsize)])

    #if self.oscilator:
    _class = torch.LongTensor([self.trigger_class]*self.batchsize)
    #else:
    #    _class = self.trgs

    #self.datas_currentepochs = [self.datas_currentepochs[i] for i in range(len(self.datas_currentepochs)) if i not in vvv_idx]
    #for _v in vvv_idx:
    #    self.datas_currentepochs.pop(_v)
    #    break

    #self.datas = self.datas[self.poison_per_batch:]
    #self.datas_currentepochs = self.datas_currentepochs[self.poison_per_batch:]
    #self.point = self.point.repeat(self.batchsize, *([1]*(len(self.point.shape)-1)))

    perm = torch.randperm(len(self.datas_currentepochs))
    vvv_idx = perm[:self.batchsize-self.poison_per_batch]
    #self.datas_currentepochs = [self.datas_currentepochs[i] for i in vvv_idx]

    point = self.point
    for i in range(self.batchsize-self.poison_per_batch):
        point[i] = self.datas_currentepochs[vvv_idx[i]][0]
        _class[i] = self.datas_currentepochs[vvv_idx[i]][1]

    self.keys = [n for n, v in self.surrogate.named_parameters()]

    #ks = torch.randperm(len(self.keys))
    #ks = ks[:7]
    #self.keys = [self.keys[i] for i in ks]

    #print(self.keys)

    self.surrogate.train()
    self.optimizer.zero_grad()
    self.surrogate.zero_grad()

    inp = point.to(device); tar = _class.to(device)
    with torch.no_grad():
        outputs = net(inp)
        base = float(sum(outputs.argmax(1) == self.trgs.to(device))) / outputs.shape[0]
        base_match = float(sum(tar == self.trgs.to(device))) / outputs.shape[0]
        acc = float(sum(outputs.argmax(1) == tar)) / outputs.shape[0]
        if verbose:
            print(f"[T] Trigger output accuracy {acc} ({base_match}) base {base}")
            print(f"[T] Poison output class {outputs[-1].argmax()}({outputs[-1].detach().cpu().numpy()}) wanted {_class[-1]}")
        saving.append(outputs[-1].detach().cpu().numpy())

    #if (epoch < 4):# or (epoch % 4 == 0):
    #if (epoch % 2 == 0):
    if (epoch < 10):
        #if epoch < 10:
        #    factor = 10
        #else:
        #    factor = 10
        factor = 1000
        perm = torch.randperm(len(self.datas_currentepochs))
        idx = perm#[:self.batchsize*factor]
        self.datas_currentepochs = [self.datas_currentepochs[i] for i in idx]
        self.notest = False
        return

    self.notest = False

    outputs = self.surrogate(inp)
    if verbose:
        print(f"[S] Poison output class {outputs[-1].argmax()}({outputs[-1].detach().cpu().numpy()}) wanted {_class[-1]}")
    loss = self.criterion(outputs, tar)
    loss.backward()

    prms = [x for x in self.surrogate.named_parameters() if x[0] in self.keys]
    vals = [v.grad.detach().clone() for n, v in sorted(prms, key=lambda x: self.keys.index(x[0]))]

    self.vvv = vals

    #-------
    # this is for baseline
    #
    #self.point = self.point.to("cpu")
    #self.datas_currentepochs = [(self.point[i], _class[i]) for i in range(self.batchsize)]
    #return
    #-------

    #-------
    # this is for no attack
    #
    #perm = torch.randperm(len(self.datas_currentepochs))[:self.batchsize]
    #self.datas_currentepochs = [self.datas_currentepochs[i] for i in perm]
    #return
    #-------

    step = 1
    best_error = torch.tensor(-1)
    best = None

    global errors

    width = 10
    depth = 2
    ops = np.random.randint(9, size=(self.poison_per_batch, width, depth))
    ws = torch.tensor(np.random.dirichlet([1]*self.poison_per_batch*width) \
                      .reshape((self.poison_per_batch, width)), requires_grad=True)
    m = torch.tensor(np.random.beta(*[[1]*self.poison_per_batch]*2), requires_grad=True)
    w = [ws, m]

    opt = torch.optim.Adam(w, lr=0.001, betas=(0.99, 0.999))
    #opt = torch.optim.SGD(w, lr=0.001)

    best_r = 1e9

    itrs = 200
    for i in range(itrs):

        _inp = torch.stack(
            [i[0] for i in self.datas_currentepochs[:self.batchsize-self.poison_per_batch]] \
          + [aug(j[0], ws[i], m[i], ops[i])
             for i, j in enumerate(
                self.datas_currentepochs[self.batchsize-self.poison_per_batch:self.batchsize]
            )]
        )
        _tar = torch.stack([self.datas_currentepochs[i][1] for i in range(self.batchsize)])

        # RANDOM SAMPLING
        perm = torch.randperm(len(self.datas_currentepochs))
        idx = perm[:self.batchsize]
        idx[:self.batchsize-self.poison_per_batch] = vvv_idx

        _inp_r = torch.stack([self.datas_currentepochs[i][0] for i in idx])
        _tar_r = torch.stack([self.datas_currentepochs[i][1] for i in idx])

        _inp_r = _inp_r.to(device); _tar_r = _tar_r.to(device)

        self.surrogate.zero_grad()
        self.surrogate.train()
        self.optimizer.zero_grad()

        outputs = self.surrogate(_inp_r)
        loss = self.criterion(outputs, _tar_r)
        loss.backward()

        prms = [x for x in self.surrogate.named_parameters() if x[0] in self.keys]
        val  = [v.grad.detach().clone() for n, v in sorted(prms, key=lambda x: self.keys.index(x[0]))]

        error_r = 0
        for x1, x2 in zip(self.vvv, val):
            error_r += torch.norm((x1-x2)/len(x1), p=2)

        best_r = min(best_r, error_r.item())

        # /RANDOM SAMPLING

        self.surrogate.zero_grad()
        self.surrogate.train()
        self.optimizer.zero_grad()

        _inp = _inp.to(device); _tar = _tar.to(device)

        if hasattr(self.optimizer, "zero_microbatch_grad"):
            self.optimizer.zero_microbatch_grad()

        outputs = self.surrogate(_inp)
        #print(f"Poison output class {outputs[-1].argmax()}({outputs[-1].detach().cpu().numpy()}) wanted {_class[-1]}")
        loss = self.criterion(outputs, _tar)
        loss.backward(create_graph=True)

        prms = [x for x in self.surrogate.named_parameters() if x[0] in self.keys]
        val  = [v.grad.clone() for n, v in sorted(prms, key=lambda x: self.keys.index(x[0]))]

        error = 0
        for x1, x2 in zip(self.vvv, val):
            error += torch.norm((x1-x2)/len(x1), p=2)

        opt.zero_grad()
        error.backward(retain_graph=i!=itrs-1, inputs=w)

        ws.grad = torch.sign(ws.grad)
        m.grad = torch.sign(m.grad)

        opt.step()

        errors.append((error.item(), best_r))

        grds = [float(v.flatten().mean()) for v in val]
        if (best is None) or (error < best_error):
            best = (_inp, _tar)
            best_error = error
            _val = val[:]
            best_mean = np.mean(grds)
            best_std = np.std(grds)
            if verbose:
                print("new best error: ", error, end="\r")
        if verbose:
            print("error: ", error.item(), "/", best_error.item(), "grad mean:", np.mean(grds), "+-", np.std(grds), end="\r")
    if verbose:
        print()
        print("Best:", best_error, "Best mean", best_mean, "+-", best_std)

    self.datas_currentepochs = list(zip(*best))

    np.save(f"errors.npy", errors)
    np.save(f"approx.npy", (_inp, inp))

    if False:
        torchvision.utils.save_image(torchvision.utils.make_grid(unnormalize(torch.stack([dc[0]
            for dc in self.datas_currentepochs]), *(MOMENTS[args.dataset])), nrow=8), "grads_imgs.png")

        torchvision.utils.save_image(torchvision.utils.make_grid(unnormalize(
            self.point, *(MOMENTS[args.dataset])), nrow=8), "grads_targets.png")
    #torchvision.utils.save_image(torchvision.utils.make_grid(self.point, nrow=8),"grads_target.png")

    if verbose:
        print("Classes: ", [x for _,x in self.datas_currentepochs])

    if False:
        plt.figure()
        bs = self.vvv[0].flatten().cpu().detach().numpy()
        plt.plot(bs, label="Target gradient", alpha=0.5)

        plt.plot(_val[0].flatten().cpu().detach().numpy(), label="Reconstruction", alpha=0.5)
        plt.grid()
        plt.xlabel(r"$\theta_i$")
        plt.ylabel("Gradient magnitude")
        plt.title("Approximation of a target gradient of Layer 1")
        plt.legend()
        plt.savefig("grads_de.png")
        plt.close()
        del _val
    return

    del candidates
    #self.mapping = dict(zip(hashes, losses))
    print()
    print(f"Indetified {len(self.mapping)} data points")

    # Sorting the values to ease the sampling later
    if verbose:
      print("Sorting ...")

    if not self.isbatchwise:
        self.datas_currentepochs.sort(key = lambda x: self.mapping[hashtensor(x[0])])
    else:
        self.datas_currentepochs.sort(key = lambda x: self.mapping[hashtensor(x[0][0])])

    if self.atype == "oscilatorout":
        self.datas_currentepochs = self.datas_currentepochs[:len(self.datas_currentepochs)//2][::-1] + self.datas_currentepochs[len(self.datas_currentepochs)//2:][::-1]

    self.datas_currentepochs = self.datas_currentepochs[:1]

  def __next__(self):
    if self.attacking:
      # If we are starting the attack phase

      if self.atype in ["oscilatorin", "oscilatorout"]:
        if self.oscilator:
            chosen = self.datas_currentepochs[-self.batchsize:]
            self.datas_currentepochs = self.datas_currentepochs[:-self.batchsize]
        else:
            chosen = self.datas_currentepochs[:self.batchsize]
            self.datas_currentepochs = self.datas_currentepochs[self.batchsize:]
        self.oscilator = not self.oscilator
      elif self.atype == "highlow":
        chosen = self.datas_currentepochs[-self.batchsize:]
        self.datas_currentepochs = self.datas_currentepochs[:-self.batchsize]
      elif self.atype == "lowhigh":
        chosen = self.datas_currentepochs[:self.batchsize]
        self.datas_currentepochs = self.datas_currentepochs[self.batchsize:]
      else:
        raise "Idk what this atype is"

      if len(chosen) == 0:
          return [None, None]

      inp, tar = map(list,zip(*chosen))
      if not self.isbatchwise:
          inp = torch.stack(inp); tar = torch.stack(tar)
      else:
          inp = inp[0]; tar = tar[0]

    else:
      # Else just return clean data
      inp, tar = next(self.iter)

      if self.batchsize is None:
        if self.isbatchwise:
          self.batchsize = 1
        else:
          self.batchsize = inp.shape[0]

      if not self.attacking:
        if self.isbatchwise:
          self.datas.append((inp, tar))
        else:
          for _inp, _tar in zip(inp, tar):
            self.datas.append((_inp, _tar))
        
      self.random = torch.rand((1, *self.datas[1][0].shape))

    # Here, we are learning of the same data that we are giving the true model to
    # get the same stage of training
    # =========
    if not self.whitebox:
        self.surrogate.train()
        self.optimizer.zero_grad()
        self.surrogate.zero_grad()

        inp = inp.to(device); tar = tar.to(device)
        outputs = self.surrogate(inp)
        loss = self.criterion(outputs, tar)
        loss.backward()
        self.optimizer.step()

    if tar.device == 'cuda':
        inp = inp.to('cpu'); tar = tar.to('cpu')
    # =========

    return (inp.detach(), tar.detach())

  def endepoch(self, epoch):
    print("Ending the batch")
    # This is called when the batcher has seen all of the data
    # Switching to the attack mode
    self.attacking = True
    # Reseting the allocation of data

    perm = torch.randperm(len(self.datas))
    idx = perm[:50000]
    self.datas_currentepochs = [self.datas[d] for d in idx]
    #self.datas_currentepochs = self.datas[:]

    # Annotating all of the data that we know of
    # to know relative complexity of the samples
    self.annotate_datas(epoch, verbose=True)

if args.adversarial:
    btch = AdversarialBatcher
else:
    btch = BaseBatcher

attackparams = {
        "atype": args.attacktype, "opt": args.surrogate_optimizer,
        "lr": args.surrogate_lr, "momentum": args.surrogate_momentum,
        "wd": args.surrogate_wd, "granularity": args.batchwise,}

if args.whitebox:
    attackparams['targetmodel'] = net
    attackparams['targetopt'] = optimizer
    attackparams['whitebox'] = True

btch = btch(trainloader, **attackparams)

overalls = []
epoch = start_epoch
while epoch < start_epoch+args.epochs:
    losses, accs = train(epoch, btch, verbose=True)
    #print("TRAIN")
    #tlosses, taccs = test(epoch, verbose=False, trainset=True, trigger=btch.trigger, trigger_class=btch.trigger_class)
    #print("TEST")
    if not btch.notest: # or (random.random() < 0.1):
        tlosses, taccs, triggers, err_triggers = test(epoch, verbose=False, trigger=btch.trigger, trigger_class=btch.trigger_class)
        overalls.append((losses, accs, tlosses, taccs, triggers, err_triggers))
        epoch += 1

    #scheduler.step()
    btch.endepoch(epoch)

with open(args.outname, "wb") as f:
    pickle.dump((args, overalls), f)