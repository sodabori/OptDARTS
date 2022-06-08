import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import utils

from args import args, beta_decay_scheduler
from model import Network, distill
from architect import Architect
from analyzer import Analyzer
from operations import NAS_BENCH_201

from perturb import Linf_PGD_alpha, Random_alpha

from copy import deepcopy
from numpy import linalg as LA

from nas_201_api import NASBench201API as API

log_name = 'search-{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), args.seed)

if args.auxiliary_skip:
    log_name += '-aux_skip'
if not args.perturb_alpha == 'none':
    log_name += '-' + str(args.perturb_alpha)
if args.unrolled:
    log_name += '-unrolled'
log_name += '-weight_optimizer-' + str(args.optimizer)
log_name += '-arch_optimizer-' + str(args.arch_optimizer)
log_name += '-' + str(np.random.randint(10000))

exp_path = os.path.join(os.path.abspath(args.exp_path), log_name)

utils.create_exp_dir(exp_path, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(exp_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.perturb_alpha == 'none':
        perturb_alpha = None
    elif args.perturb_alpha == 'pgd_linf':
        perturb_alpha = Linf_PGD_alpha
    elif args.perturb_alpha == 'random':
        perturb_alpha = Random_alpha
    
    api = API(os.path.abspath(args.bench_data))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion)  # N=5/1/3
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.get_weights(),
            args.sgd_learning_rate,
            momentum=args.sgd_momentum,
            weight_decay=args.sgd_weight_decay)
    
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.get_weights(),
            args.adam_learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay)
    
    else:
        raise Exception("Invalid weight optimizer: check argument --optimizer")

    if args.arch_optimizer == 'sgd':
        arch_optimizer = torch.optim.SGD(
            model.arch_parameters(),
            lr=args.arch_sgd_learning_rate,
            momentum=args.arch_sgd_momentum,
            weight_decay=args.arch_sgd_weight_decay)
        
    elif args.arch_optimizer == 'adam':
        arch_optimizer = torch.optim.Adam(model.arch_parameters(),
            lr=args.arch_adam_learning_rate,
            betas=(args.arch_adam_beta1, args.arch_adam_beta2),
            weight_decay=args.arch_adam_weight_decay)

    else:
        raise Exception("Invalid architecture optimizer: check argument --arch_optimizer")


    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split_train = int(np.floor(args.train_portion * num_train))
    split_valid = int(np.floor((1 - args.train_portion) * num_train))
    print('num_train =', num_train, 'split_train =', split_train, 'split_valid =', split_valid)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split_train]),
        pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split_valid:num_train]),
        pin_memory=True)

    if args.optimizer == 'sgd':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.sgd_learning_rate_min)
    elif args.optimizer == 'adam':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.adam_learning_rate_min)

    analyzer = Analyzer(model, args)
    architect = Architect(model, optimizer, arch_optimizer, args.unrolled)

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]

        if args.cutout:
            # increase the cutout probability linearly throughout search
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e beta %e', epoch, lr, beta_decay_scheduler.decay_rate)
        
        if args.perturb_alpha:
            epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.epochs
            logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(model.show_alphas())

        # training
        train_acc, train_obj, ev = train(train_queue, valid_queue, model, architect, criterion, optimizer, arch_optimizer, lr, 
                                         perturb_alpha, epsilon_alpha, epoch, analyzer)
        logging.info('train_acc %f ev %f', train_acc, ev)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        # nasbench201
        result = api.query_by_arch(model.genotype(), hp='200')  # hp='200'
        logging.info('{:}'.format(result))
        cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
            cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)

        state = {
            "epoch": epoch,
            "Acc/train": train_acc,
            "Loss/train": train_obj,
            "Analysis/eigenvalue: ": ev,
            "Acc/valid": valid_acc,
            "Loss/valid": valid_obj,
            "NAS-Bench-201/cifar10/train": cifar10_train,
            "NAS-Bench-201/cifar10/test": cifar10_test,
            "NAS_Bench-201/cifar100/train": cifar100_train,
            "NAS-Bench-201/cifar100/valid": cifar100_valid,
            "NAS-Bench-201/cifar100/test": cifar100_test,
            "NAS-Bench-201/imagenet16/train": imagenet16_train,
            "NAS-Bench-201/imagenet16/valid": imagenet16_valid,
            "NAS-Bench-201/imagenet16/test": imagenet16_test
        }
        
        # Architecture parameter infomations

        softmax_alphas = model.normalized_arch_parameters()

        idx2edge = {v: k for k, v in model.edge2index.items()}

        for i, edge in idx2edge.items():
            for j, op_name in enumerate(NAS_BENCH_201):
                state[f"Architecture/{edge}/{op_name}"] = softmax_alphas[i][j].item()

        if epoch == args.epochs - 1:
            utils.save_checkpoint(model, optimizer, args.exp_path, log_name)

        scheduler.step()
        beta_decay_scheduler.step()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, arch_optimizer, lr, perturb_alpha, epsilon_alpha, epoch, analyzer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, eta=lr)
        optimizer.zero_grad()
        architect.arch_optimizer.zero_grad()

        # print('before softmax', model.arch_parameters())
        model.softmax_arch_parameters()

        # perturb on alpha
        # print('after softmax', model.arch_parameters())
        if perturb_alpha:
            perturb_alpha(model, input, target, epsilon_alpha)
            optimizer.zero_grad()
            arch_optimizer.zero_grad()
        # print('after perturb', model.arch_parameters())

        logits = model(input, updateType='weight')
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.restore_arch_parameters()
        # print('after restore', model.arch_parameters())

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    _data_loader = deepcopy(train_queue)
    input, target = next(iter(_data_loader))

    input = input.cuda()
    target = target.cuda(non_blocking=True)

    H = analyzer.compute_Hw(input, target, input_search, target_search, lr, optimizer, False)

    del _data_loader

    ev = max(LA.eigvals(H.cpu().data.numpy()))
    ev = np.linalg.norm(ev)

    return  top1.avg, objs.avg, ev


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 