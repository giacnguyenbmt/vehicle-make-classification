import argparse
import io
import os
from collections import OrderedDict

import cv2
import numpy as np
import subprocess
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from mss import mss
from PIL import Image

NUM_CLASS = 19

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(
    description='PyTorch Vehicle Make Classification Training'
)
parser.add_argument('data', metavar='DIR', nargs='?', default='input.jpg',
                    help='path to input image (default: input.jpg)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--augment', action='store_true', help="use data augmentation")
parser.add_argument('--print-model', action='store_true', help="print model")


def main():
    args = parser.parse_args()
    model, inference_transform, device = main_worker(args)

    with open(args.data, 'rb') as f:
        image_bytes = f.read()
        tensor = transform_image(image_bytes, inference_transform, device)
    with torch.no_grad():
        class_name = get_prediction(tensor, model)
    print(class_name)

def infer(model, image):
    ...

def main_worker(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if args.augment:
        print("Data augmentation is used!")
        inference_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        print("Data augmentation is NOT used!")
        inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASS)
    if args.print_model:
        print(f"Model arch: {model}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # optionally resume from a checkpoint
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

        if torch.cuda.is_available():
            model.to(device)

        # Check model device
        print(
            "=> Device which model is using:", 
            next(model.parameters()).device
        )
        
        if checkpoint.get('epoch', None) is None:
            print("=> loaded checkpoint '{}'".format(args.checkpoint))
        else:
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.checkpoint, checkpoint['epoch']))

        model.eval()

        return model, inference_transform, device
        
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        return None, None, None
    
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

def mainsct(trafficlight):  
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')    

    sct = mss()

    crop = {'top': 0, 'left': 0, 'width': int(resolution[0]), 'height': int(resolution[1])}
    # result = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    # _,frame = cap.read()
    frame = np.array(sct.grab(crop))
    bbox = cv2.selectROI(frame)

    bbox_coords = [bbox]

    i = 0
    while(True):
        # ret, frame = cap.read()
        frame = np.array(sct.grab(crop))
        # if ret == False:
        #     print('Video is ended')
        #     break
        
        labels, confs = trafficlight.predict(frame, bbox_coords)
        frame = trafficlight.draw(frame, bbox_coords, labels, confs)

        # result.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        print(f'Done Frame{i}')
        i += 1

if __name__ == '__main__':
    main()