import os
import gc
import sys
import time
import copy
import numpy as np
import shutil
import warnings
import random
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models


import filters

# List of References:

# Model:
# https://github.com/pytorch/examples/blob/main/imagenet/main.py
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# https://github.com/nugglet/CV-50.035-PSETS/blob/main/PSET%202/week6/libs/solver.py
# https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/

# top-k accuracy:
# https://stats.stackexchange.com/questions/95391/what-is-the-definition-of-top-n-accuracy
# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b

# gradient accumulation:
# https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3

# performance tuning:
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# channels_last: https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/6f7327daa2a9b857365f893069d0bace/memory_format_tutorial.ipynb#scrollTo=Ybj_JWb3XWaL


class CNNClassifier(object):
    def __init__(self, model_name: str, train_data_path: str, alias: str, **kwargs):

        # Required arguments:
        # - model: A model object conforming to the API described above
        # - data: path to training data folder
        # - alias: file name to save model weights

        self.model_name = model_name
        self.alias = alias # model name to save checkpoint file

        # get classes
        self.classes, self.class_dict = self._find_classes(train_data_path)

        # handling gpu
        if torch.cuda.is_available():
            print('CUDA available')
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
            
        else:
            print('WARNING: CUDA is not available')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.lr = kwargs.pop('lr', 1e-1)
        self.lr_decay = kwargs.pop('lr_decay', 1e-4)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.accumulation = kwargs.pop('accumulation', 4)
        self.num_epochs = kwargs.pop('num_epochs', 90)
        self.start_epoch = kwargs.pop('start_epoch', 0) #'manual epoch number (useful on restarts)'
        self.momentum = kwargs.pop('momentum', 0.9)
        self.seed = kwargs.pop('seed', 42)
        self.workers = kwargs.pop('workers', 4)

        self.resume = kwargs.pop('resume', None) # path to checkpoint file
        self.feature_extracting = kwargs.pop('feature_extracting', False)
  
        self.image_size = kwargs.pop('image_size', 256)
        self.colour_distort = kwargs.pop('colour_distort', False)
        self.gaussian_noise = kwargs.pop('gaussian_noise', False)
        self.gaussian_blur = kwargs.pop('gaussian_blur', False)

        self.top_k = kwargs.pop('top_k', 3)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.print_model = kwargs.pop('print_model', False)

        # get data
        props = [ transforms.Resize(self.image_size+1),
                    transforms.CenterCrop(self.image_size)
                    ]

        # apply data augmentation
        if self.colour_distort:
            props.append(*filters.get_color_distortion())
        if self.gaussian_blur:
            props.append(*filters.get_gaussian_blur())

        props.append(transforms.ToTensor())
        if self.gaussian_noise:
            props.append(*filters.get_gaussian_noise())
            
        self.transform = transforms.Compose(props)

        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        full = datasets.ImageFolder(train_data_path, transform=self.transform)

        # split dataset into train and test sets
        dataset_total_size = len([f for f in Path(train_data_path).glob('*\*.JPEG')])
        train, test = torch.utils.data.random_split(full, [int(dataset_total_size * 0.8), dataset_total_size - int(dataset_total_size * 0.8)], generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True)
        test_dataloader = DataLoader(test, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True)
        
        # load val data into dataloader if path given. No data augmentation is performed on the validation data.
        self.val_data_path = kwargs.pop('val_data_path', None)
        if self.val_data_path is not None:
            val_props = transforms.Compose([transforms.Resize(self.image_size+1),
                                            transforms.CenterCrop(self.image_size), 
                                            transforms.ToTensor()])
            val = datasets.ImageFolder(self.val_data_path, transform=val_props)
            val_dataloader = DataLoader(val, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
        else:
            val_dataloader = None

         # put datasets into a dictionary
        self.dataloaders = {'train': train_dataloader, 'test': test_dataloader, 'val': val_dataloader}

        # init model
        if model_name == 'resnet':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        elif model_name == 'alexnet':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

        if self.feature_extracting:
            self._set_parameter_requires_grad(self.model, self.feature_extracting)
      
        self.model = self.model.to(device=self.device)

        if self.print_model:
            print(self.model.eval())

        # initialize criterion, optimizer and scheduler
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.lr_decay)

        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)

        # if resuming training from checkpoint
        if self.resume:
            if os.path.isfile(self.resume):
                print(f"=> loading checkpoint '{self.resume}'")
                checkpoint = torch.load(self.resume)
            
            self.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint '{self.resume}' (epoch {checkpoint['epoch']})")

        else:
            print(f"=> no checkpoint found at '{self.resume}'")

        # Print Model
        print(f'Created model: {self.model_name} with pre-processing filter.')

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)


    def train(self):
        
        best_acc1 = 0
        start = time.time()

        #  Autotuner runs a short benchmark and selects the kernel with the best performance on a given hardware for a given input size.
        cudnn.benchmark = True

        # training loop
        for epoch in range(self.start_epoch, self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # train for one epoch
            epoch_start = time.time()
            self._train(epoch)

            # evaluate on test set
            acc1 = self._test()
            self.scheduler.step()
            
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            self._save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.model_name,
                'state_dict': self.model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : self.optimizer.state_dict(),
                'scheduler' : self.scheduler.state_dict()
            }, is_best, alias=self.alias)

            print(f"Epoch {epoch}: completed in {(time.time() - epoch_start)/60 :.2f} minutes")
        print(f"Training Complete in {(time.time() - start)/60 :.2f} minutes")
        gc.collect()
        torch.cuda.empty_cache()
        
    def validate(self):
        # Make predictions on validation set (custom dataset) and return accuracy and results
        if self.dataloaders['val'] is None:
            return "No validation dataset specified"

        else:
            val_loader = self.dataloaders['val']
            batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
            losses = AverageMeter('Loss', ':.4e', Summary.NONE)
            top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
            top5 = AverageMeter(f'Acc@{self.top_k}', ':6.2f', Summary.AVERAGE)
            progress = ProgressMeter(
                len(val_loader),
                [batch_time, losses, top1, top5],
                prefix='Validate: ')

            # switch to evaluate mode
            self.model.eval()

            with torch.no_grad():
                end = time.time()
                for i, (images, target) in enumerate(val_loader):
            
                    # if torch.cuda.is_available():
                    #     target = target.cuda(0, non_blocking=True)
                    images = images.to(self.device)
                    target = target.to(self.device)

                    # compute output
                    output = self.model(images)
                    loss = self.criterion(output, target)

                    # measure accuracy and record loss
                    acc_1, acc_k = accuracy(output, target, topk=(1, self.top_k))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc_1[0], images.size(0))
                    top5.update(acc_k[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % self.print_every == 0:
                        progress.display(i)

                progress.display_summary()

            return top1.avg
        

    def _train(self, epoch):
        train_loader = self.dataloaders['train']
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter(f'Acc@{self.top_k}', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        self.model.train()

        end = time.time()
        self.optimizer.zero_grad()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # if torch.cuda.is_available():
            #     target = target.cuda(0, non_blocking=True)
            images = images.to(self.device)
            target = target.to(self.device)

            # compute output
            output = self.model(images) # forward step
            loss = self.criterion(output, target)
            loss = loss / self.accumulation # normalize since CE loss is averaged

            # compute gradient and do SGD step
            loss.backward()

            # clear graph every n steps
            if (i+1) % self.accumulation == 0:
                self.optimizer.step() 
                self.optimizer.zero_grad()

                 # measure accuracy and record loss
                acc_1, acc_k = accuracy(output, target, topk=(1, self.top_k))
                losses.update(loss.item(), images.size(0))
                top1.update(acc_1[0], images.size(0))
                top5.update(acc_k[0], images.size(0))
        

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_every == 0:
                progress.display(i)

    def _test(self):

        test_loader = self.dataloaders['test']
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter(f'Acc@{self.top_k}', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(test_loader):
        
                # if torch.cuda.is_available():
                #     target = target.cuda(0, non_blocking=True)
                images = images.to(self.device)
                target = target.to(self.device)

                # compute output
                output = self.model(images)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc_1, acc_k = accuracy(output, target, topk=(1, self.top_k))
                losses.update(loss.item(), images.size(0))
                top1.update(acc_1[0], images.size(0))
                top5.update(acc_k[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_every == 0:
                    progress.display(i)

            progress.display_summary()

        return top1.avg

    def _save_checkpoint(self, state, is_best, alias='cnn'):
        filename=f'./checkpoints/{alias}_checkpoint.pth.tar'
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, f'./checkpoints/{alias}_model_best.pth.tar')

    def _set_parameter_requires_grad(self):
        # This helper function sets the .requires_grad attribute of the parameters in the model to False when we are feature extracting. 
        # By default, when we load a pretrained model all of the parameters have .requires_grad=True, which is fine if we are training from scratch or finetuning. 
        # However, if we are feature extracting and only want to compute gradients for the newly initialized layer then we want all of the other parameters 
        # to not require gradients.
        for param in self.model.parameters():
            param.requires_grad = False

    def _find_classes(self, dir):
        classes = os.listdir(dir)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

