import os
from tqdm import tqdm
import torch
from utils import Bar, Logger, AverageMeter, accuracy, savefig
import torch.optim as optim
import time
import torch.nn.functional as F

class Learner():
    def __init__(self,model,args,trainloader, testloader, class_per_task, logit_init_ind, use_cuda, path, train_path, infer_path):
        self.model=model
        self.args=args
        self.title= self.args['dataset'] + self.args['arch']
        self.trainloader=trainloader 
        self.use_cuda=use_cuda
        self.testloader=testloader
        self.start_epoch=self.args['start_epoch']
        self.test_loss=0.0
        self.path = path
        self.train_path = train_path
        self.infer_path = infer_path
        self.test_acc=0.0
        self.train_loss, self.train_acc=0.0,0.0
        self.mapped_labels = range(class_per_task)
        self.class_per_task = class_per_task
        self.logit_init_ind = logit_init_ind

        trainable_params = []

        if (self.args['arch'] in ["resnet50", "resnet18"]):
            params_set = [self.model.convs1, self.model.bns1, self.model.layers1, self.model.layers2,
                          self.model.layers3, self.model.layers4]

        for j, params in enumerate(params_set):
            if j > 0:
                j -= 1
            for i, param in enumerate(params):
                if isinstance(param, list):
                    param = param[args['test_case']]
                if (self.train_path[j, i] == 1):
                    p = {'params': param.parameters()}
                    trainable_params.append(p)
                else:
                    param.requires_grad = False

                    
        p = {'params': self.model.fc.parameters()}
        trainable_params.append(p)
        print("Number of layers being trained : " , len(trainable_params))

#         self.optimizer = optim.Adadelta(trainable_params)
#         self.optimizer = optim.SGD(trainable_params, lr=self.args['lr'], momentum=0.96, weight_decay=0)
        self.optimizer = optim.SGD(trainable_params, lr=self.args['lr'], weight_decay=0.0, momentum=0.9,
                                   nesterov=True)
        



    def learn(self):
        if self.args['resume']:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(self.args['resume']), 'Error: no checkpoint directory found!'
            self.args['checkpoint'] = os.path.dirname(self.args['resume'])
            checkpoint = torch.load(self.args['resume'])
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(self.args['checkpoint'], 'log_'+str(self.args['task_num'])+'_'+str(self.args['test_case'])+'.txt'), title=self.title, resume=True)
        else:
            logger = Logger(os.path.join(self.args['checkpoint'], 'log_'+str(self.args['task_num'])+'_'+str(self.args['test_case'])+'.txt'), title=self.title)
            logger.set_names(['Sess', 'Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        if self.args['evaluate']:
            print('\nEvaluation only')
            self.test(self['start_epoch'])
            print(' Test Loss:  %.8f, Test Acc:  %.2f' % (self.test_loss, self.test_acc))
            return

        for epoch in range(self.args['start_epoch'], self.args['num_epochs']):
            self.adjust_learning_rate(epoch)

            print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch + 1, self.args['num_epochs'], self.args['lr'],self.args['task_num']))
            self.train(epoch, self.infer_path)
            self.test(epoch, self.infer_path)
            # append logger file
            logger.append([self.args['task_num'], epoch, self.args['lr'], self.train_loss, self.test_loss, self.train_acc, self.test_acc])

            # save model
            if epoch % self.args['log_freq'] == 0:
                self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'acc': self.test_acc,
                        'optimizer' : self.optimizer.state_dict(),
                }, checkpoint=self.args['savepoint'], task_num=self.args['task_num'], test_case=self.args['test_case'])

        logger.close()
        logger.plot()
        savefig(os.path.join(self.args['checkpoint'], 'log_'+str(self.args['task_num'])+'_'+str(self.args['test_case'])+'.eps'))


    def train(self, epoch, path):
        # switch to train mode
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.trainloader)):
            # measure data loading time
            data_time.update(time.time() - end)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs, path, self.args['task_num'])
            tar_ce = targets
            pre_ce = outputs.clone()

            pre_ce = pre_ce[:, self.logit_init_ind:self.logit_init_ind+self.class_per_task]

            loss = F.cross_entropy(pre_ce, tar_ce)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data[:, self.logit_init_ind:self.logit_init_ind + self.class_per_task],targets.data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix  = '({batch}/{size}) | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} '.format(
                        batch=batch_idx + 1,
                        size=len(self.trainloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.train_loss,self.train_acc=losses.avg, top1.avg

   
    
    def test(self, epoch, path):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        bar = Bar('Processing', max=len(self.testloader))
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.testloader)):
            # measure data loading time
            data_time.update(time.time() - end)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            outputs = self.model(inputs, path, self.args['task_num'])

            tar_ce = targets
            pre_ce = outputs.clone()

            pre_ce = pre_ce[:, self.logit_init_ind:self.logit_init_ind+self.class_per_task]

            loss = F.cross_entropy(pre_ce, tar_ce)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data[:, self.logit_init_ind:self.logit_init_ind + self.class_per_task],targets.data, topk=(1, 5))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(self.testloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                        )
            bar.next()
        bar.finish()
        self.test_loss= losses.avg;self.test_acc= top1.avg

    def save_checkpoint(self,state, checkpoint='checkpoint', filename='checkpoint.pth.tar',task_num=0, test_case=0):
        torch.save(state, os.path.join(checkpoint, 'session_'+str(task_num)+'_'+str(test_case)+'_model.pth.tar'))

    def adjust_learning_rate(self, epoch):
        if epoch in self.args['schedule']:
            self.state['lr'] *= self.args['gamma']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args['lr']

    def get_confusion_matrix(self, path):

        confusion_matrix = torch.zeros(self.class_per_task, self.class_per_task)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloader):
                if self.args['use_cuda']:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.model(inputs, path, self.args["task_num"])
                pre_ce = outputs.clone()

                pre_ce = pre_ce[:,self.logit_init_ind:self.logit_init_ind+self.class_per_task]
                _, preds = torch.max(pre_ce, 1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        print(confusion_matrix)
        return confusion_matrix

