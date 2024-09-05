import os
from typing import List, Dict, Type
import math

import torch
from torch.optim import Optimizer
import transformers
import matplotlib.pyplot as plt

from data.gen_brain_qa import generate_adhd_qa_txt
from sklearn import metrics

def adhd_eval(model, eval_dataloader, dataset_type, gpu_id=0):
    print ('start adhd_eval ...')
    eval_data_iterator = iter(eval_dataloader)
    num_iter = len(eval_dataloader)

    pred_label = []
    gt_label = []
    for eval_iter in range(num_iter):           
        eval_data = next(eval_data_iterator)
        images = eval_data['images'].cuda().half()
        
        adhd_answer = []        ### 0 healthy  1 ADHD
        adhd_tq = []

        bs = len(eval_data['reports'])
        for b in range(bs):
            report = eval_data['reports'][b]
            # print(b, ': ', report)
            if dataset_type == 'adhd':
                adhd_doc = generate_adhd_qa_txt(report)
                
            adhd_answer.append(adhd_doc['answer'])
            adhd_tq.append(adhd_doc['tq'])

        # print (adhd_tq)
        model.eval()
        adhd_res = model.generate({"images": images, 'prompt': adhd_tq}, device='cuda:{}'.format(gpu_id)) # "images": images,

        for i in range(bs):
            pos = adhd_res[i].find('[SEP]')
            if pos >= 0:
                pred_answer = adhd_res[i][:pos-1]
            else:
                pred_answer = adhd_res[i]

            if 'Yes' in pred_answer:
                pred_label.append(1)
            else:
                pred_label.append(0)

            gt_answer = adhd_answer[i][:-6]
            if 'Yes' in gt_answer:
                gt_label.append(1)
            else:
                gt_label.append(0)
    
    acc = metrics.accuracy_score(gt_label, pred_label)
    recall = metrics.recall_score(gt_label, pred_label)
    precision = metrics.precision_score(gt_label, pred_label)
    f1 = metrics.f1_score(gt_label, pred_label)
    kappa = metrics.cohen_kappa_score(gt_label, pred_label)

    print ('eval acc = {:.3f}, f1 = {:.3f}, recall = {:.3f}, precision = {:.3f}, kappa = {:.3f}'.format(acc, f1, recall, precision, kappa))
    return acc, f1, recall, precision, kappa

class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        pass

    def train(self,
        model,
        train_data,
        dataloader,
        eval_dataloader,
        gpu_id,
        dataset_type,
        epochs: int = 1,
        scheduler_name: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        output_path: str = './checkpoints/vision_text_pretrain',
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''

        print ('enter train process ...')
        print ('accumulation_steps = ', accumulation_steps)
        
        self.accumulation_steps = accumulation_steps
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler_name=scheduler_name, warmup_steps=warmup_steps, t_total=num_train_steps)

        model = model.cuda()

        train_loss_set = []
        best_acc = 0
        best_f1 = 0
        best_kappa = 0
        best_epoch = 0

        if not os.path.exists(output_path): 
            os.makedirs(output_path)
        valid_log = os.path.join(output_path, 'model_log.txt')
        fp = open(valid_log, 'w')

        skip_scheduler = False
        for epoch in range(epochs):
            print ('start {}th epoch training ...'.format(epoch))
            # dataloader.sampler.set_epoch(epoch)       ### multi-gpu
            data_iterator = iter(dataloader)

            for train_iter in range(steps_per_epoch):
                model.zero_grad()
                model.train()              
                data = next(data_iterator)

                if use_amp:
                    with autocast():
                        loss = model(data)
                    loss_value = loss['loss']
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss = model(data)
                    # loss_value = loss['loss'] / self.accumulation_steps
                    loss_value = loss['loss']
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                if train_iter % 10 == 0:
                    print('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch,epochs, train_iter, steps_per_epoch, loss_value))
                    # train_loss_set.append(loss_value)
                
                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()
            
            ### model validation stage
            if (epoch+1) > 10 and (epoch+1) % 5 == 0: 
                eval_acc, eval_f1, eval_recall, eval_precision, eval_kappa = adhd_eval(model, eval_dataloader, dataset_type, gpu_id=gpu_id)
                fp.write('epoch {}: acc {:.3f}, f1 {:.3f}, recall {:.3f}, precision {:.3f}, kappa {:.3f}\n'.format(epoch, eval_acc, eval_f1, eval_recall, eval_precision, eval_kappa))
                
                if best_kappa <= eval_kappa:
                    best_kappa = eval_kappa
                    best_acc = eval_acc
                    best_f1 = eval_f1
                    best_epoch = epoch
                    self._save_ckpt(model, epoch+1, output_path)
                    print ('save best model epoch {}, acc {:.3f}, f1 {:.3f}, kappa {:.3f}'.format(epoch, best_acc, best_f1, best_kappa))
        
        fp.write('best_epoch = {}, best_acc = {:.3f}, best_f1 = {:.3f}, kappa {:.3f}'.format(best_epoch, best_acc, best_f1, best_kappa))
        fp.close()

    @staticmethod
    def _get_scheduler(optimizer, scheduler_name: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler_name = scheduler_name.lower()   
        if scheduler_name == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler_name == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler_name == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler_name == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler_name == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler_name))

    def _save_ckpt(self, model, epoch, save_dir, model_name='best_model.pth'):
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        state_dict = model.state_dict()
        
        torch.save(state_dict, os.path.join(save_dir, model_name))
