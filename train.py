from model import ClassificationModel
from datasets import Dataset
import numpy as np
import pandas as pd
import random
import json
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse

def tuplify_with_device(batch, device):
    return tuple([batch[0].to(device, dtype=torch.long), batch[1].to(device, dtype=torch.long),
                  batch[2].to(device, dtype=torch.float), batch[3].to(device, dtype=torch.float)])                  

def compute_accuracy(y_pred, y_target):
    correct = accuracy_score(y_target, y_pred.round())    
    def_roc = roc_auc_score(y_target, y_pred)
    return correct, def_roc

def set_seed(seed):
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def main(args, train_loss_set, val_loss_set):
    train_loss_set = []
    argument_path = f'{args.save}/model{args.sa_num}.json'
    argument = {'seed' : args.seed, 'learning_rate' : args.learning_rate, \
            'batch_size' : args.batch_size, 'epoch' : args.num_epochs, 'max_grad_norm' : args.max_grad_norm, \
            'factor' : args.factor, 'mode' :args.sched_mode, 'patience' : args.patience}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    set_seed(args.seed)
    model = ClassificationModel()
    model.to(device)

    #clip model freeze    
    model.clip_freeze() # clip laeyr weight all freeze

    #Data load
    train = pd.read_json(f"{args.train_path}/train.json")
    val = pd.read_json(f"{args.val_path}/val.json")
    
    #train data loader
    train_data = Dataset(train, args.image_path)
    train_dataloader =  DataLoader(train_data, batch_size = args.batch_size)

    #val data loader
    val_data = Dataset(val, args.image_path)
    val_dataloader =  DataLoader(val_data, batch_size = args.batch_size)


    loss_func = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode=args.sched_mode, factor=args.factor,
                                           patience=args.patience, verbose = True)
    
    best_ep = 0
    best_model_state_on_dev = None
    best_dev_acc = 0.0
    best_dev_auroc = 0.0
    
    for e_idx in range(1,args.num_epochs+1):
        print('Epoch {}/{}'.format(e_idx, args.num_epochs))
        print('----------------------')

        tr_loss = 0.0
        nb_tr_steps = 0
    
        model.train()
    # train
        for batch_index, batch in enumerate(tqdm(train_dataloader)):
            batch = tuplify_with_device(batch, device)

            b_input_ids, b_attention_mask,\
            b_pixel_values, b_labels = batch

            optimizer.zero_grad()

            y_pred = model(input_ids = b_input_ids, attention_mask = b_attention_mask,
                             pixel_values = b_pixel_values)
            loss = loss_func(y_pred.view(-1, 1), b_labels.view(-1, 1))
            loss_item = loss.item()
            train_loss_set.append(loss_item)
            

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
       
            tr_loss += loss_item
            nb_tr_steps += 1
            
        print("train loss: {:.4f}".format(tr_loss / nb_tr_steps))

        val_loss = 0.0
        nb_val_steps = 0
    # validation
        dev_y_preds, dev_y_targets = [], []
        model.eval()       
        for batch_index, batch in enumerate(val_dataloader):
            batch = tuplify_with_device(batch, device)
            b_input_ids, b_attention_mask,\
            b_pixel_values, b_labels = batch
            with torch.no_grad():
                y_pred = torch.sigmoid(model(input_ids = b_input_ids, attention_mask = b_attention_mask, pixel_values = b_pixel_values))

            val_loss = loss_func(y_pred.view(-1, 1), b_labels.view(-1, 1))
            val_loss_item = val_loss.item()
            val_loss_set.append(val_loss_item)

            y_pred = y_pred.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            dev_y_preds.append(y_pred)
            dev_y_targets.append(label_ids)

            val_loss += val_loss_item
            nb_val_steps += 1

        scheduler.step(loss)
            
        dev_y_preds = np.concatenate(dev_y_preds).reshape((-1, ))
        dev_y_targets = np.concatenate(dev_y_targets).reshape((-1, )).astype(int)
        dev_acc, dev_auroc = compute_accuracy(dev_y_preds, dev_y_targets)
        
        if dev_acc > best_dev_acc: 
            best_ep = e_idx
            best_dev_acc = dev_acc
            best_dev_auroc = dev_auroc
            best_model_state_on_dev = model.state_dict()       

        print("val loss: {:.4f}".format(val_loss / nb_val_steps))
        print("val acc: {:.4f}".format(dev_acc))
        print("val auroc: {:.4f}".format(dev_auroc))
        print()

    
    print("best eq: {:.4f}".format(best_ep))
    print("best_dev_acc: {:.4f}".format(best_dev_acc))
    print("best_dev_auroc: {:.4f}".format(best_dev_auroc))
    torch.save(best_model_state_on_dev, f'{args.save}/model{args.sa_num}.pt') 
    
    argument['train_loss'] = train_loss_set
    argument['best_ep'] = best_ep
    argument['best_dev_acc'] = best_dev_acc
    argument['best_dev_auroc'] = best_dev_auroc
    with open(argument_path, 'w') as f:
        json.dump(argument, f)


if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument('--seed', default=1337, help='random seed', type=int)
    parser.add_argument('--learning_rate', default=0.001, help='learning_rate', type=float)
    parser.add_argument('--batch_size', default=128, help='batch_size', type=int)
    parser.add_argument('--num_epochs', default=10, help='epoch', type=int)
    parser.add_argument('--max_grad_norm', default=1.0, help='clip_grad_norm parameter', type=float)
    parser.add_argument('--factor', default=0.5, help='scheduler parameter', type=float)
    parser.add_argument('--sched_mode', default='min', help='scheduler parameter', type=str)
    parser.add_argument('--patience', default=1, help='scheduler parameter',type=int)
    parser.add_argument('--save', default='model_pt', help='model save path',type=str)
    parser.add_argument('--sa_num', default='1', help='model save number',type=str)
    parser.add_argument('--image_path', default='image', help='image path',type=str)
    parser.add_argument('--train_path', default='.', help='train data path',type=str)
    parser.add_argument('--val_path', default='.', help='val data path',type=str)
    args = parser.parse_args()

    train_loss_set = []
    val_loss_set = []
    main(args, train_loss_set, val_loss_set)