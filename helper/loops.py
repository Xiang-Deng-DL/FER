from __future__ import print_function, division

import sys
import time
import torch
from .util import AverageMeter, accuracy

import torch.nn.functional as F


def train_vanilla(epoch, train_loader, model, criterion, criterion_soft, optimizer, opt, train_soft, train_index):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    if torch.cuda.is_available():
        train_soft=train_soft.cuda()
    
    for idx, (input, target, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        
        if epoch==opt.init:
            
            _, predicted = torch.max(output.detach(), 1)
            
            current_t = predicted == target
            
            current_t = current_t.type(torch.ByteTensor).cuda()
            
            train_index[index] = train_index[index] | current_t
            
            soft_label = criterion_soft(output.detach())
            
            probs = F.softmax( output.detach(), dim=1 )
            
            ture_batch = (current_t == 1).nonzero(as_tuple=False).view(-1)
            
            if len(ture_batch) > 0:
                
                soft_label_true = soft_label[ture_batch]
            
                x_axis = torch.range(0, len(target)-1 )
            
                x_axis = x_axis.long()
            
                cur_pros = probs[ [x_axis, target.long()]  ]
            
                cur_pros_ture = cur_pros[ture_batch].view(-1,1)
                
                cur_pros_ture = 1 - torch.exp(-cur_pros_ture*opt.mu)
            
                index_true = index[ture_batch]
                
                train_soft[index_true] = train_soft[index_true]*(1 - cur_pros_ture) + soft_label_true*cur_pros_ture
               
        
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()


        # print info
        if idx>0 and idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.4f}'.format(top1=top1, top5=top5, losses = losses))

    return top1.avg, train_soft, train_index



def train_ol(epoch, train_loader, model, criterion_list, optimizer, opt, train_soft, train_index):
    
    # set modules as train()
    model.train()

    if torch.cuda.is_available():
        train_soft=train_soft.cuda()

    criterion_cls = criterion_list[0]
    critetion_soft = criterion_list[1]
    criterion_kl = criterion_list[2]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    for idx, data in enumerate(train_loader):
        input, target, index = data

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            soft_targets = train_soft[index]
            batch_index = train_index[index]
            batch_index = batch_index.float()

            ture_bat = (batch_index == 1).nonzero(as_tuple=False).view(-1)
            false_bat = (batch_index == 0).nonzero(as_tuple=False).view(-1)

        # ===================forward=====================
        
        preact = False

        logit_s = model(input, is_feat=False, preact=preact)
        
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        
        loss_cls_true = torch.sum( loss_cls[ture_bat] )
        loss_cls_false = torch.sum( loss_cls[false_bat] )
        
        loss_kl_true = criterion_kl(logit_s[ture_bat], soft_targets[ture_bat])
        
        loss = ( (opt.gamma+(1-(0.0+epoch)/opt.epochs)*(1-opt.gamma))*loss_cls_true + (0.0+epoch)/opt.epochs*(1-opt.gamma)*loss_kl_true + loss_cls_false )/len(target)

        # update
        _, predicted = torch.max(logit_s.detach(), 1)
        
        current_t = predicted == target
        
        current_t = current_t.type(torch.ByteTensor).cuda()
            
        train_index[index] = train_index[index] | current_t
        
        soft_label = critetion_soft(logit_s.detach())
        
        probs = F.softmax(logit_s.detach(), dim=1)
        
        ture_batch = (current_t == 1).nonzero(as_tuple=False).view(-1)
        
        if len(ture_batch) > 0:
            
            soft_label_true = soft_label[ture_batch]
            
            x_axis = torch.range(0, len(target)-1 )
        
            x_axis = x_axis.long()
            
            cur_pros = probs[ [x_axis, target.long()]  ]
            
            cur_pros_ture = cur_pros[ture_batch].view(-1,1)
            
            cur_pros_ture = 1 - torch.exp(-cur_pros_ture*opt.mu)
            
            index_true = index[ture_batch]
                
            train_soft[index_true] = train_soft[index_true]*(1 - cur_pros_ture) + soft_label_true*cur_pros_ture


        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx>0 and idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {losses.avg:.4f}'.format(top1=top1, top5=top5, losses = losses))
    

    return top1.avg, train_soft, train_index




def validate(val_loader, model, criterion, opt, past):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target, index) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()

            # compute output
            output = model(input)
            
            _, predicted = torch.max(output, 1)

            loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            
            batch_time.update(time.time() - end)
            end = time.time()

            if idx>0 and idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print('Test Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


