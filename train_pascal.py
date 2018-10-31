import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MODELS.model_resnet import *
from PIL import ImageFile
import datasets, logger
import pdb
import cv2
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', default=50, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--ngpu', default=4, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
parser.add_argument('--out_dir', type=str, default='test/')
best_prec1 = 0



CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)

    saved_dir = os.path.join(args.out_dir, 'checkpoints')
    log_dir = os.path.join(args.out_dir, 'log')
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    logger.configure(dir=log_dir)

    # Data loading code
#    traindir = os.path.join(args.data, 'train')
#    valdir = os.path.join(args.data, 'val')
    traindir = args.data
    valdir = args.data
    train_list = './VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt'
    val_list = './VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # import pdb
    # pdb.set_trace()
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_list, transforms.Compose([
               # transforms.Scale(256),
               # transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
                ])),
            batch_size=args.batch_size, shuffle=False,
           num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_dataset = datasets.ImageFolder(
        traindir, train_list,
        transforms.Compose([
            #transforms.RandomSizedCrop(size0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None
    print(args.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    # create model
    if args.arch == "resnet":
        model = ResidualNet( 'ImageNet', args.depth, 20, args.att_type )

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print ("model")
    print (model)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))




    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True
    best_acc =0
    saved_value = (0,0,0,0,0,0, torch.cuda.FloatTensor(np.zeros(20)))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch)
        acc, saved_value = train(train_loader, model, multi_class_cross_entropy_loss, optimizer, epoch, saved_value)
        
        # evaluate on validation set it does not needed
        #prec1 = validate(val_loader, model, criterion, epoch)
        #prec1 = validate(val_loader, model, multi_class_cross_entropy_loss, epoch)
        
        # remember best prec@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(best_acc, acc)
        #best_prec1 = max(prec1, best_prec1)
        save_checkpoint(saved_dir, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'acc':acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, epoch+1)

def multi_class_cross_entropy_loss(preds, labels, eps=1e-6):
    cls_loss = labels * torch.log(preds +eps) + (1-labels) * torch.log(1-preds +eps)
    summed_cls_loss = torch.sum(cls_loss, dim=1)
    loss = -torch.mean(summed_cls_loss, dim=0)
    if torch.isnan(loss.sum()) :
        pdb.set_trace()
    return loss


def train(train_loader, model, criterion, optimizer, epoch, saved_value):
#    batch_time = 0#AverageMeter()
#    data_time = 0#AverageMeter()
#    losses = 0#AverageMeter()
#    all_accs = 0#AverageMeter()
#    cls_accs = 0#AverageMeter()
#    cnt = 0
    (batch_time, data_time, losses, all_accs, cls_accs, cnt,cls_cnt) = saved_value

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        #pdb.set_trace()
        #img = input[0].cpu().numpy() + np.array([[[102.9801, 115.9465, 122.7717]]]).reshape(3,1,1)
        #img = np.ascontiguousarray(np.transpose(img,(1,2,0)))
        #cv2.imwrite('images.png',img)
        cnt += 1
        # measure data loading time
        data_time += (time.time() - end)
        
        target = target.cuda() #target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        all_acc, cls_acc, cls_cnt = pascal_accuracy(output.data, target,cls_cnt)
        #prec1, prec5 = pascal_accuracy(output.data, target, topk=(1, 5))
        #pdb.set_trace()
        losses += loss
        all_accs += all_acc
        cls_accs += cls_acc
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time += time.time() - end
        end = time.time()
        cnt 
        if i % args.print_freq == 0:
            abs_batch_time = batch_time / cnt
            abs_data_time = data_time /cnt
            abs_losses = losses.item() / cnt
            abs_all_accs = all_accs/cnt
            abs_all_accs = abs_all_accs.item()
            abs_cls_accs = cls_accs / cls_cnt
            abs_cls_accs[abs_cls_accs != abs_cls_accs] = 0
            #abs_all_accs = all_accs.item() / cnt
            logger.log('Epoch: [{}][{}/{}]\t Time {}\t Data {}\t Loss {}\t All acs {} '.format(
                   epoch, i, len(train_loader), abs_batch_time,
                   abs_data_time, abs_losses, abs_all_accs))
            
            logger.log((cls_accs/(cnt)))
            print(cls_cnt)
            
            logger.record_tabular('loss',loss.item())
            logger.record_tabular('loss_accum', abs_losses)
            logger.record_tabular('accum_all_acces', abs_all_accs)
            temp = output[0] >= 0.5
            print("PRED",end=' ')
            for i in range(cls_acc.shape[0]):
                if temp[i].item() : 
                    print(CLASSES[i],end=' ')
            print("\t\t\tGT",end=' ')
            for i in range(cls_acc.shape[0]):
                if target[0,i]  == 1: 
                    print(CLASSES[i],end=' ')
            print()

            for i in range(cls_accs.shape[0]):
                logger.record_tabular('accum_cls_accs_{:02d}'.format(i), abs_cls_accs[i].item()/(cnt))
                logger.record_tabular('cls_accs_{:02d}'.format(i), cls_acc[i].item())


            logger.dump_tabular()
            
    return all_accs.item()/cnt, (batch_time, data_time, losses, all_accs, cls_accs, cnt, cls_cnt)


def validate(val_loader, model, criterion, epoch):
    batch_time = 0#AverageMeter()
    data_time = 0#AverageMeter()
    losses = 0#AverageMeter()
    all_accs = 0#AverageMeter()
    cls_accs = 0#AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda() #(async=True)
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        all_acc, cls_acc = pascal_accuracy(output.data, target)
       # prec1, prec5 = pascal_accuracy(output.data, target, topk=(1, 5))
        losses += loss
        all_accs += all_acc
        cls_accs += cls_acc
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            abs_batch_time = batch_time / (i+1)
            abs_data_time = data_time / (i+1)
            abs_losses = losses.item() / (i+1)
            abs_all_accs = all_accs.item() / (i+1)
            logger.log('Epoch: [{}][{}/{}]\t Time {}\t Data {}\t Loss {}\t All acs {} '.format(
                   epoch, i, len(train_loader), abs_batch_time,
                   abs_data_time, abs_losses, abs_all_accs))
            
            logger.log((cls_accs/(i+1)))
            
            logger.record_tabular('val/loss',loss.item())
            logger.record_tabular('val/accum_loss', abs_losses)
            logger.record_tabular('val/accum_all_acces', abs_all_accs)
            for i in range(cls_accs.shape[0]):
                logger.record_tabular('val/accum_cls_accs_{}'.format(i), cls_accs[i].item()/(i+1))
                logger.record_tabular('val/cls_accs_{}'.format(i), cls_acc[i].item())

            logger.dump_tabular()
        
    return all_accs.item()/(i+1)


def save_checkpoint(saved_dir,state, is_best, prefix):
    filename=os.path.join(saved_dir,'{:04d}.pth'.format(prefix))
    torch.save(state, filename)
    print("SAVE checkpoint {}".format(filename))
    if is_best:
        print("BEST")
        shutil.copyfile(filename, os.path.join(saved_dir, 'model_best_{:04d}.pth'.format(prefix)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def pascal_accuracy(output, target, cls_cnt) :
    pred = output >= 0.5
    acc = pred.type(torch.cuda.FloatTensor) * target
    acc = acc.type(torch.cuda.FloatTensor)
    all_acc = acc.mean(dim=1)
    cls_acc = acc / target
    cls_acc[cls_acc != cls_acc] = 0 # NaN to 0

    cls_cnt += target.sum(dim=0)
    cls_acc = cls_acc / target.sum(dim=0)
    cls_acc[cls_acc != cls_acc] = 0 
    return all_acc.mean(dim=0), cls_acc, cls_cnt

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pdb.set_trace()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
