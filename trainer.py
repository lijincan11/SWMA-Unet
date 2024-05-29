import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
from medpy.metric.binary import dc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from torchinfo import summary
def plot_result(dice, h, snapshot_path,args):
    dict = {'mean_dice': dice, 'mean_hd95': h} 
    df = pd.DataFrame(dict)
    plt.figure(0)
    df['mean_dice'].plot()
    resolution_value = 1200
    plt.title('Mean Dice')
    date_and_time = datetime.datetime.now()
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    plt.savefig(save_mode_path, format="png", dpi=resolution_value)
    plt.figure(1)
    df['mean_hd95'].plot()
    plt.title('Mean hd95')
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png'
    save_mode_path = os.path.join(snapshot_path, filename)
    #save csv 
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv'
    save_mode_path = os.path.join(snapshot_path, filename)
    df.to_csv(save_mode_path, sep='\t')

def inference(model, testloader, args, test_save_path=None):
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info(' idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    metric_list = metric_list / len(testloader.dataset)
    
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    return performance, mean_hd95

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    
    test_save_path = os.path.join(snapshot_path, 'test')
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    print(summary(model, input_size=(batch_size, 3, 224, 224)))
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    dice_=[]
    hd95_= []
    Max_dice = 0.80 

    loss_history = []
    fig,ax = plt.subplots()
    for epoch_num in iterator:
        running_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            running_loss += loss.item()
            try:
                if iter_num % 20 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)
            except: pass
        
        # Test args.eval_interval
        if (epoch_num + 1) % 20 == 0 and (epoch_num + 1)>149:
        # if (epoch_num + 1) % 2 == 0 :
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            if mean_dice > Max_dice:
                filename = f'{args.model_name}_epoch_{epoch_num}.pth'
                save_mode_path = os.path.join(snapshot_path, filename)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num}")
                print(f"Epoch {epoch_num}")
                Max_dice = mean_dice
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()


        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        logging.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch_num+1, max_epoch, avg_loss))

        # 绘制当前训练损失
        plt.figure()
        ax = plt.gca()
        ax.plot(loss_history)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(os.path.join(snapshot_path, f"loss.png"))
        plt.close()

        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                print(f"Epoch {epoch_num}, Last Epcoh")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()

            iterator.close()
            break
    plot_result(dice_,hd95_,snapshot_path,args)
    writer.close()
    return "Training Finished!"

def trainer_ACDC(args, model, snapshot_path):
    from datasets.dataset_ACDC import ACDC_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # add validation and test set #
    db_val=ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="valid")
    db_test =ACDC_dataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader=DataLoader(db_val, batch_size=1, shuffle=False)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    test_save_path = os.path.join(snapshot_path, 'test')
    
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    loss_history = []
    fig,ax = plt.subplots()
    iter_num = 0
    avg_hd = 100.0
    Loss = []
    Test_Accuracy = []
    dice_=[]
    hd95_= []

    Best_dcs = 0.90
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s   %(levelname)s   %(message)s')

    for epoch_num in iterator:
        running_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            running_loss += loss.item()

            try:
                if iter_num % 20 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image('train/Image', image, iter_num)
                    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)
            except: pass
        avg_loss = running_loss / len(trainloader)
        loss_history.append(avg_loss)
        logging.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch_num+1, max_epoch, avg_loss))

        # 绘制当前训练损失
        plt.figure()
        ax = plt.gca()
        ax.plot(loss_history)
        ax.set_title('Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.savefig(os.path.join(snapshot_path, f"loss.png"))
        plt.close()
        
        # Test
        if (epoch_num + 1) % args.eval_interval == 0 and (epoch_num+1)>139:
        # if (epoch_num + 1) % 2 == 0:
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            if mean_dice > Best_dcs:
                filename = f'{args.model_name}_epoch_{epoch_num}.pth'
                save_mode_path = os.path.join(snapshot_path, filename)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num}")
                print(f"Epoch {epoch_num}")
                Best_dcs = mean_dice
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            model.train()

        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                print(f"Epoch {epoch_num}, Last Epcoh")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()
                
            iterator.close()
            break
    plot_result(dice_,hd95_,snapshot_path,args)
    writer.close()
    return "Training Finished!"