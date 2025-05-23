from __future__ import print_function

import argparse
import os
import shutil
import time
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training")
# Datasets
parser.add_argument("-d", "--dataset", default="cifar10", type=str)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
# Optimization options
parser.add_argument(
    "--epochs", default=164, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--train-batch", default=128, type=int, metavar="N", help="train batchsize"
)
parser.add_argument(
    "--test-batch", default=100, type=int, metavar="N", help="test batchsize"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--drop",
    "--dropout",
    default=0,
    type=float,
    metavar="dropout",
    help="dropout ratio",
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[81, 122],
    help="decrease learning rate at these epochs.",
)
parser.add_argument(
    "--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule."
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
# Checkpoints
parser.add_argument(
    "-c",
    "--checkpoint",
    default="checkpoints",
    type=str,
    metavar="PATH",
    help="path to save checkpoint (default: checkpoints)",
)
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: None)",
)
parser.add_argument(
    "--use-timestamp", action="store_true", help="whether to use timestamp"
)
# Architecture
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet)",
)
parser.add_argument("--depth", type=int, default=164, help="Model depth.")
parser.add_argument(
    "--block-name",
    type=str,
    default="bottleneck",
    help="the building block for Resnet and Preresnet: basicBlock, bottleneck (default: bottleneck for cifar10/cifar100)",
)
parser.add_argument(
    "--module",
    default=None,
    type=str,
    help="attention module",
)
parser.add_argument(
    "--use-asr", action="store_true", help="whether to use ASR"
)
parser.add_argument(
    "--use-both", action="store_true", help="whether to use ASR"
)
# Miscs
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument(
    "--e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
# Device options
parser.add_argument(
    "--gpu-id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert (
    args.dataset == "cifar10" or args.dataset == "cifar100"
), "Dataset can only be cifar10 or cifar100."

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    
    args.checkpoint = os.path.join(args.checkpoint, f"{args.dataset}-{args.arch}{args.depth}")
    
    if args.module is not None:
        args.checkpoint = f"{args.checkpoint}-{args.module}"
        
    if args.use_asr:
        args.checkpoint = f"{args.checkpoint}-asr"
    
    if args.use_both:
        args.checkpoint = f"{args.checkpoint}-both"
    
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.checkpoint = f"{args.checkpoint}-{timestamp}"
        
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print("==> Preparing dataset %s" % args.dataset)
    
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]
    )
    if args.dataset == "cifar10":
        dataloader = datasets.CIFAR10
        num_classes = 10
    if args.dataset == "cifar100":
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(
        root="../datasets", train=True, download=True, transform=transform_train
    )
    testset = dataloader(
        root="../datasets", train=False, download=False, transform=transform_test
    )
    trainloader = data.DataLoader(
        trainset,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
    )
    testloader = data.DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers
    )

    # Model
    print("==> creating model '{}{}'".format(args.arch, args.depth))
    if args.arch.endswith("resnet"):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            block_name=args.block_name,
            module=args.module,
            use_asr=args.use_asr,
            use_both=args.use_both
        )
    else:
        raise Exception(
            args.arch
            + " is not defined, please check whether the model name is correct"
        )

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print(
        "    Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )
    print(
        "    Total params: %.8fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    title = f"{args.dataset}_{args.arch}{args.depth}_{args.module}"
    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger = Logger(
            os.path.join(args.checkpoint, "log.txt"), title=title, resume=True
        )
    else:
        logger = Logger(os.path.join(args.checkpoint, "log.txt"), title=title)
        logger.set_names(
            ["Learning Rate", "Train Loss", "Valid Loss", "Train Acc.", "Valid Acc."]
        )

    if args.evaluate:
        print("\nEvaluation only")
        for m in model.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy(delete=True)
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(" Test Loss:  %.8f, Test Acc:  %.2f" % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, args.epochs, state["lr"]))

        for m in model.modules():
            if hasattr(m, 'switch_to_train'):
                m.switch_to_train()
        train_loss, train_acc = train(
            trainloader, model, criterion, optimizer, epoch, use_cuda
        )
        
        for m in model.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state["lr"], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "acc": test_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint=args.checkpoint,
        )
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, "log.eps"))

    print("Best acc:")
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(
            targets
        )

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        print(suffix)
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(
                targets
            )

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            print(suffix)

    return (losses.avg, top1.avg)


def save_checkpoint(
    state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state["lr"] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = state["lr"]


if __name__ == "__main__":
    main()
