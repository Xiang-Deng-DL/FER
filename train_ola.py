from __future__ import print_function

import os
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, set_seed
from helper.loops import train_vanilla as train1, train_ol as train2, validate

from distiller_zoo import Softmax_T, KL

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1600, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=240, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')  
    
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed') # in most experiments, seeds 1,5, 7 are used

    # dataset
    parser.add_argument('--model', type=str, default='ResNet18', choices=['wrn_40_2', 'MobileNetV2', 'ShuffleV2', 'ResNet18'])

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    
    parser.add_argument('-r', '--gamma', type=float, default=0.1, help='minimum weight for CE, typically set to 0.1')
    
    parser.add_argument('-i', '--init', type=int, default=1, help='initial epochs if needed, here simply set it to 1')

    parser.add_argument('--mu', type=float, default=0.9, help='mu')
     
    parser.add_argument('--kd_T', type=float, default=5, help='temperature for soft behavior')
    
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')    
    

    opt = parser.parse_args()
    
    if opt.model in ['MobileNetV2', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/models'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    
    best_acc = 0

    opt = parse_option()
    
    print(opt)
    
    
    if opt.seed is not None:
        set_seed(opt.seed)

    # dataloader
    if opt.dataset == 'cifar100':
        
        print('cifar100')
                
        n_cls = 100
        
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    criterion_cls = nn.CrossEntropyLoss(reduce = False)    
    
    criterion_soft = Softmax_T(opt.kd_T)
    
    criterion_kl = KL(opt.kd_T)
    

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)   # classification loss
    criterion_list.append(criterion_soft)
    criterion_list.append(criterion_kl)



    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    
    train_soft = torch.zeros((50000,100), dtype=torch.float).cuda()
    
    train_index = torch.zeros(50000)
    
    train_index = train_index.type(torch.ByteTensor).cuda()
    
    train_probs = torch.zeros(50000)
    
    train_probs = train_probs.float().cuda()
    
    train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, 
                                                                is_instance=True,  is_shuffle=True)
    
    past = torch.zeros(10000)
    
    past = past.type(torch.ByteTensor).cuda()
    
    for epoch in range(1, opt.epochs + 1):
        
        adjust_learning_rate(epoch, opt, optimizer)

        print("==> training...")

        time1 = time.time()
        
        if epoch<=opt.init:
            train_acc, train_soft, train_index = train1(epoch, train_loader, model, criterion, criterion_soft, optimizer, opt, train_soft, train_index)
        else:
            train_acc, train_soft, train_index = train2(epoch, train_loader, model, criterion_list, optimizer, opt, train_soft, train_index)
        
        time2 = time.time()

        
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_acc, test_acc_top5  = validate(val_loader, model, criterion, opt, past)


        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)
        

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)


    # This best accuracy is only for printing purpose.
    # The results reported in the paper is from the last epoch.
    print('best accuracy:', best_acc.cpu().numpy() )
    

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)
    

if __name__ == '__main__':
    main()
