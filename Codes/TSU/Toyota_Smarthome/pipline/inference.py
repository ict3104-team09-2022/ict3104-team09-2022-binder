from __future__ import division

import argparse
import time
import torch
import csv
import os
import sys
import warnings
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd


warnings.filterwarnings("ignore")
from tqdm import tqdm

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-rgb_model_file', type=str)
parser.add_argument('-flow_model_file', type=str)
parser.add_argument('-gpu', type=str, default='4')
parser.add_argument('-dataset', type=str, default='charades')
parser.add_argument('-rgb_root', type=str, default='no_root')
parser.add_argument('-flow_root', type=str, default='no_root')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.1')
parser.add_argument('-epoch', type=str, default='5')
parser.add_argument('-model', type=str, default='PDAN_TSU_RGB')
parser.add_argument('-APtype', type=str, default='map')
parser.add_argument('-randomseed', type=str, default='False')
parser.add_argument('-load_model', type=str, default='./models/PDAN_TSU_RGB')
parser.add_argument('-num_channel', type=str, default='512')
parser.add_argument('-batch_size', type=str, default='False')
parser.add_argument('-kernelsize', type=str, default='False')
parser.add_argument('-feat', type=str, default='False')
parser.add_argument('-split_setting', type=str, default='CS')
parser.add_argument('-VideoInputTest', type=str, default='P02T01C06', help='input video file name')
parser.add_argument('-video_path', type=str, default='False', help='input video file path')
parser.add_argument('-featurePath', type=str)
parser.add_argument('-annotations', type=str)
parser.add_argument('-filepath', type=str)

args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import cv2
import math

# set random seed
if args.randomseed == "False":
    SEED = 0
elif args.randomseed == "True":
    SEED = random.randint(1, 100000)
else:
    SEED = int(args.randomseed)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('Random_SEED!!!:', SEED)

from torch.autograd import Variable

if str(args.APtype) == 'map':
    from apmeter import APMeter

batch_size = int(args.batch_size)

if args.dataset == 'TSU':
    filename = ""
    split_setting = str(args.split_setting)

    from smarthome_i3d_per_video import TSU as Dataset
    from smarthome_i3d_per_video import TSU_collate_fn as collate_fn

    classes = 51

    if split_setting == 'CS':
        train_split = args.rgb_root
        # test_split = './data/smarthome_CS_51.json'
        test_split = args.rgb_root

    elif split_setting == 'CV':
        train_split = './data/smarthome_CV_51.json'
        test_split = './data/smarthome_CV_51.json'

    # rgb_root = '/data/stars/user/rdai/smarthome_untrimmed/features/i3d_16frames_64000_SSD'
    rgb_root = args.featurePath
    flow_root = r""
    skeleton_root = '/skeleton/feat/Path/'

# Activity List for Inference
activityList = ["Enter", "Walk", "Make_coffee", "Get_water", "Make_Coffee",
                "Use_Drawer", "Make_coffee.Pour_grains", "Use_telephone",
                "Leave", "Put_something_on_table", "Take_something_off_table", "Pour.From_kettle", "Stir_coffee/tea",
                "Drink.From_cup", "Dump_in_trash", "Make_tea",
                "Make_tea.Boil_water", "Use_cupboard", "Make_tea.Insert_tea_bag", "Read", "Take_pills", "Use_fridge",
                "Clean_dishes", "Clean_dishes.Put_something_in_sink",
                "Eat_snack", "Sit_down", "Watch_TV", "Use_laptop", "Get_up", "Drink.From_bottle", "Pour.From_bottle",
                "Drink.From_glass",
                "Lay_down", "Drink.From_can", "Write", "Breakfast", "Breakfast.Spread_jam_or_butter",
                "Breakfast.Cut_bread", "Breakfast.Eat_at_table", "Breakfast.Take_ham",
                "Clean_dishes.Dry_up", "Wipe_table", "Cook", "Cook.Cut", "Cook.Use_stove",
                "Cook.Stir", "Cook.Use_oven", "Clean_dishes.Clean_with_water",
                "Use_tablet", "Use_glasses", "Pour.From_can"]


def video_inference(models, num_epochs=50):
    probs = []
    for model, gpu, dataloader, optimizer, sched, model_file in models:
        prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], 0)
        probs.append(prob_val)
        sched.step(val_loss)
        arrayForMaxAndIndex = []
        try:
            for index in range(len(prob_val.get(fileName)[1])):
                activityAtEachFrameArray = []
                for index1 in range(len(prob_val.get(fileName))):
                    activityAtEachFrameArray.append(prob_val.get(fileName)[index1][index])
                maxValue = max(activityAtEachFrameArray)
                indexOfMaxValue = activityAtEachFrameArray.index(maxValue)
                arrayForMaxAndIndex.append([activityList[indexOfMaxValue], maxValue])
            create_caption_video(arrayForMaxAndIndex)
        except TypeError:
            print("File does not belong in the testing dataset CS_32.")



  


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data_rgb_skeleton(train_split, val_split, root_skeleton, root_rgb):
    # Load Data

    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root_skeleton, root_rgb, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 8
    else:
        dataset = None
        dataloader = None

    val_dataset = Dataset(val_split, 'testing', root_skeleton, root_rgb, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)  # 2

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    print(datasets)

    return dataloaders, datasets


def load_data(train_split, val_split, root):
    # Loading Data for training
    if len(train_split) > 0:
        dataset = Dataset(train_split, 'training', root, batch_size, classes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None


    #Loading data for testing
    val_dataset = Dataset(val_split, 'testing', root, batch_size, classes)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=collate_fn)
    val_dataloader.root = root

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    return dataloaders, datasets


# train the model
def run(models, criterion, num_epochs=50):
    since = time.time()

    best_map = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        probs = []
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            # train_map, train_loss = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], epoch)
            probs.append(prob_val)
            sched.step(val_loss)

            # probs.append(train_map)
            # sched.step(train_loss)

            # value = 10
            if best_map < val_map:
                best_map = val_map
                # best_map = value
                torch.save(model.state_dict(),
                            args.filepath + str(args.model) + '/weight_epoch_' + str(args.lr) + '_' + str(epoch))
                torch.save(model, args.filepath + str(args.model) + '/model_epoch_' + str(args.lr) + '_' + str(epoch))
                print('save here:', args.filepath + str(args.model) + '/weight_epoch_' + str(args.lr) + '_' + str(epoch))


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    # print(results)
    print("Total Evaluation Results: " + str(len(results)))
    print(results.get("P02T01C06"))
    return results


def run_network(model, data, gpu, epoch=0, baseline=False):
    inputs, mask, labels, other = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    mask_list = torch.sum(mask, 1)
    mask_new = np.zeros((mask.size()[0], classes, mask.size()[1]))
    for i in range(mask.size()[0]):
        mask_new[i, :, :int(mask_list[i])] = np.ones((classes, int(mask_list[i])))
    mask_new = torch.from_numpy(mask_new).float()
    mask_new = Variable(mask_new.cuda(gpu))

    inputs = inputs.squeeze(3).squeeze(3)
    activation = model(inputs, mask_new)

    outputs_final = activation

    if args.model == "PDAN":
        # print('outputs_final1', outputs_final.size())
        outputs_final = outputs_final[:, 0, :, :]
    # print('outputs_final',outputs_final.size())
    outputs_final = outputs_final.permute(0, 2, 1)
    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    loss_f = F.binary_cross_entropy_with_logits(outputs_final, labels, size_average=False)
    loss_f = torch.sum(loss_f) / torch.sum(mask)

    loss = loss_f

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return outputs_final, loss, probs_f, corr / tot


def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data
        loss.backward()
        optimizer.step()
    if args.APtype == 'wap':
        train_map = 100 * apm.value()
    else:
        train_map = 100 * apm.value().mean()
    print('train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0

    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, epoch)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs = probs.squeeze()

        full_probs[other[0][0]] = probs.data.cpu().numpy().T

    epoch_loss = tot_loss / num_iter

    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    print('val-map:', val_map)
    print(100 * apm.value())
    apm.reset()

    return full_probs, epoch_loss, val_map


def create_caption_video(arrayWithCaptions):
    video = filePath
    print("video is: ", video)
    cap = cv2.VideoCapture(video)
    print("cap is: ", cap)
    print("Len", len(arrayWithCaptions))
    print("No: ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    numberOfFramePerCaption = math.ceil(length / len(arrayWithCaptions))
    print("numberOfFramePerCaption: ", numberOfFramePerCaption)

    # Get video dimensions = height, weight and frames per second
    video_fps = cap.get(cv2.CAP_PROP_FPS),
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 480.0
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 640.0
    output = args.filepath

    # we are using avc1 codec for mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output + '/'+ f'{fileName}' + '_caption.mp4', apiPreference=0, fourcc=fourcc,
                             fps=video_fps[0], frameSize=(int(width), int(height + 100)))

    pbar = tqdm(total=length)
    i = 0  # frame counter
    counter = 0  # counter for arrayWithCaptions
    index = 0  # Pointer to indicate which event begins and ends in the start frames and end frames
    events = csvFileRead()
    current_position_annotation = 0

    while True:
        # Capture frames in the video
        ret, frame = cap.read()
        # describe the type of font to be used.

        # Add white background for video inference captions
        image = cv2.copyMakeBorder(frame, 100, 0, 0, 0, cv2.BORDER_CONSTANT, None, value=(100,100,0))

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Use putText() method for
        # inserting text on video
        cv2.putText(image,
                    "Ground Truth:",
                    (200, int(height - 460)),
                    font, 0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,
                    "Prediction Value:",
                    (10, int(height - 460)),
                    font, 0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4)

        cv2.putText(image,
                    "Number of Frames:",
                    (450, int(height - 460)),
                    font, 0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4)

        cv2.putText(image,
                    str(i),
                    (500, int(height - 420)),
                    font, 0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_4)

        caption = arrayWithCaptions[counter][0] + " " + str(round(arrayWithCaptions[counter][1], 2))
        try:
            if i % numberOfFramePerCaption == 0:
                if counter < len(arrayWithCaptions) - 1:
                    counter += 1
                    caption = arrayWithCaptions[counter][0] + " " + str(round(arrayWithCaptions[counter][1], 2))
        except ZeroDivisionError:
            print("Please ensure the video file is in the data folder!")

        # overlay captions on the frame with background (image)
        cv2.putText(image,
                    caption,
                    (10, int(height - 420)),
                    font, 0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_4)
        # Overlay ground truth captions (current event provided by the annotation csv file)
        if int(events[current_position_annotation][1]) <= i <= int(
                events[current_position_annotation][2]):
            event = events[current_position_annotation][0]
            cv2.putText(image,
                        event,
                        (200, int(height - 390)),
                        font, 0.5,
                        (0, 0, 0),
                        2,
                        cv2.LINE_4)
            # Handling if there are multiple events in the same frame
            if current_position_annotation < len(events) - 1:
                if int(events[current_position_annotation + 1][1]) <= i <= int(
                        events[current_position_annotation + 1][2]):
                    event2 = events[current_position_annotation + 1][0]
                    cv2.putText(image,
                                event2,
                                (200, int(height - 390)),
                                font, 0.5,
                                (0, 0, 0),
                                2,
                                cv2.LINE_4)

        # If frame is more than or equal to  end frame of the event, move to the next event and it is not the last event
        if i >= int(events[current_position_annotation][2]) and current_position_annotation < len(events) - 1:
            # Go to next event
            current_position_annotation += 1

        # Show the progress bar
        pbar.set_description(f"Generating video:.... {i}")
        pbar.update(1)

        i += 1
        writer.write(image)

        # Uncomment to display the external video player frame
        # cv2.imshow('video', image)
        # writer.write(image)

        # creating 'q' as the quit
        # button for the video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if not ret:
            break
        if i > length:
            break

    writer.release()
    # release the cap object
    cap.release()
    # close all windows
    cv2.destroyAllWindows()

    print('Video Inference Processing complete!')




def csvFileRead():
    data = []
    annotations_directory_list = list()
    for root, dirs, files in os.walk(args.annotations, topdown=False):
        for name in dirs:
            annotations_directory_list.append(os.path.join(root, name))

    edited_annotations_directory_list = list()
    for dir in annotations_directory_list:
        strip = args.annotations + "\\"
        edited_annotations_directory_list.append(dir.lstrip(strip))

    for dir in edited_annotations_directory_list:
        if dir == fileName[:3]:
            with open(args.annotations + "/" + dir + '/' + str(fileName + '.csv'), 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
                data.pop(0)
            break
    return data



if __name__ == '__main__':
    print(str(args.model))
    print('batch_size:', batch_size)
    print('cuda_avail', torch.cuda.is_available())

    fileName = args.VideoInputTest
    # Remove .mp4 from fileName
    fileName = fileName[:-4]
    print("File name: ", fileName)
    filePath = args.video_path
    if filePath[0] == '\'':
        filePath = filePath[1:-1]
    print("FilePath: ", filePath)
    print("Pre-trained Model: ", args.model)
    print("Model is Loaded", args.load_model)


    if args.mode == 'flow':
        print('flow mode', flow_root)
        dataloaders, datasets = load_data(train_split, test_split, flow_root)
    elif args.mode == 'skeleton':
        print('Pose mode', skeleton_root)
        dataloaders, datasets = load_data(train_split, test_split, skeleton_root)
    elif args.mode == 'rgb':
        print('RGB mode', rgb_root)
        #load data with train_split (80% number of workers is 8) use test_split for testing, 20% number of workers is 2
        dataloaders, datasets = load_data(train_split, test_split, rgb_root)
    

    # if args.train is False:
    #     num_channel = args.num_channel
    #     if args.mode == 'skeleton':
    #         input_channnel = 256
    #     else:
    #         input_channnel = 1024

    #     num_classes = classes
    #     mid_channel = int(args.num_channel)

    #     if args.model == "PDAN":
    #         print("you are processing PDAN")
    #         from models import PDAN as Net

    #         model = Net(num_stages=1, num_layers=5, num_f_maps=mid_channel, dim=input_channnel, num_classes=classes)

    #     model = torch.nn.DataParallel(model)
        
    

    if args.train is False:
        num_channel = args.num_channel
        if args.mode == 'skeleton':
            input_channnel = 256
        else:
            input_channnel = 1024

        num_classes = classes
        mid_channel = int(args.num_channel)

        if args.model == "PDAN":
            print("you are processing PDAN")
            from models import PDAN as Net

            model = Net(num_stages=1, num_layers=5, num_f_maps=mid_channel, dim=input_channnel, num_classes=classes)

        model = torch.nn.DataParallel(model)

        

      
        if args.load_model != "False":
            # entire model
            model = torch.load(args.load_model)
            # weight
            # model.load_state_dict(torch.load(str(r"content/drive/MyDrive/Colab/ICT3104/PDAN/")))
            print("loaded", args.load_model)
            

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        print('num_channel:', num_channel, 'input_channnel:', input_channnel, 'num_classes:', num_classes)
        model.cuda()

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        print(lr)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, verbose=True)
        # run([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_epochs=int(args.epoch))
        video_inference([(model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], num_epochs=int(args.epoch))

        


