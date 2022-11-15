import argparse
import os
import io
import time
from enum import Enum
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

NUM_CLASS = 19

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='PyTorch Vehicle Make Classification Training'
)
parser.add_argument('data', metavar='DIR', nargs='?', default='val',
                    help='path to val dataset (default: val)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--augment', action='store_true', help="use data augmentation")
parser.add_argument('--print-model', action='store_true', help="print model")


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if args.augment:
        print("Data augmentation is used!")
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        print("Data augmentation is NOT used!")
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    valdir = args.data
    val_dataset = datasets.ImageFolder(
        valdir, val_transform
    )

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False, sampler=val_sampler)
    class_names = val_dataset.classes


    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, NUM_CLASS)
    model.fc = nn.Linear(num_ftrs, len(class_names))
    if args.print_model:
        print(f"Model arch: {model}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    

    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))

        if torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint)
        else:
            # if torch.backends.mps.is_available():
            #     loc = 'mps'
            # else:
            loc = 'cpu'
            print("=> Load location:", loc)
            checkpoint = torch.load(args.checkpoint, map_location=loc)
        model.load_state_dict(remove_module_in_state_dict(checkpoint['state_dict']))

        if not torch.cuda.is_available():
            model.to(device)
        # Check model device
        print(
            "=> Device which model is using:", 
            next(model.parameters()).device
        )

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.checkpoint, checkpoint['epoch']))


        validate()

        # model.eval()

        # with open(args.data, 'rb') as f:
        #     image_bytes = f.read()
        #     tensor = transform_image(image_bytes, inference_transform, device)
        # with torch.no_grad():
        #     class_name = get_prediction(tensor, model)
        # print(class_name)
        
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        return
    
def transform_image(image_bytes, inference_transform, device):
    image = Image.open(io.BytesIO(image_bytes))
    image = inference_transform(image).unsqueeze(0).to(device)
    return image

def get_prediction(tensor, model):
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return pred2name(predicted_idx)

def pred2name(pred):
    pred2id = {
        0: '0',
        1: '1',
        2: '10',
        3: '11',
        4: '12',
        5: '13',
        6: '14',
        7: '15',
        8: '16',
        9: '17',
        10: '18',
        11: '2',
        12: '3',
        13: '4',
        14: '5',
        15: '6',
        16: '7',
        17: '8',
        18: '9'
    }
    id2name = {
        '0': 'Audi',
        '1': 'BMW',
        '2': 'Chevrolet',
        '3': 'Ford',
        '4': 'Honda',
        '5': 'KIA',
        '6': 'Land Rover',
        '7': 'Lexus',
        '8': 'MINI',
        '9': 'Mazda',
        '10': 'Mercedes-Benz',
        '11': 'Mitsubishi',
        '12': 'NISSAN',
        '13': 'Peugeot',
        '14': 'Porsche',
        '15': 'Subaru',
        '16': 'Toyota',
        '17': 'Volkswagen',
        '18': 'Volvo',
    }
    return id2name[pred2id[pred]]

def remove_module_in_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name_ = 'module.' # remove `module.`
        if k.startswith(name_):
            new_state_dict[k[len(name_):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def validate(val_loader, model, criterion, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                # if torch.backends.mps.is_available():
                #     images = images.to('mps')
                #     target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader), # + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return top1.avg, losses.avg


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
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

if __name__ == '__main__':
    main()