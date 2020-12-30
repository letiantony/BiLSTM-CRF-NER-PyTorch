# encoding:utf-8
import os
import time
import numpy as np
import torch
from ..callback.progressbar import ProgressBar
from ..utils.utils import AverageMeter
from .train_utils import restore_checkpoint, model_device
from .metrics import Entity_Score
from .train_utils import batchify_with_label
from .metrics import F1_score


# 训练包装器
class Trainer(object):
    def __init__(self, model,
                 train_data,
                 val_data,
                 optimizer,
                 epochs,
                 logger,
                 evaluate,
                 avg_batch_loss=False,
                 label_to_id=None,
                 n_gpu=None,
                 lr_scheduler=None,
                 resume=None,
                 model_checkpoint=None,
                 training_monitor=None,
                 early_stopping=None,
                 writer=None,
                 verbose=1,
                 bioes=False,
                 sep='_',
                 train_loader=None,
                 num_classes=-1):
        self.model = model  # 模型
        self.train_loader = train_loader
        self.train_data = train_data  # 训练数据
        self.val_data = val_data  # 验证数据
        self.epochs = epochs  # epochs次数
        self.optimizer = optimizer  # 优化器
        self.logger = logger  # 日志记录器
        self.verbose = verbose  # 是否打印
        self.writer = writer  # tensorboardX写入器
        self.training_monitor = training_monitor  # 监控训练过程指标变化
        self.early_stopping = early_stopping  # early_stopping
        self.resume = resume  # 是否重载模型
        self.model_checkpoint = model_checkpoint  # 模型保存
        self.lr_scheduler = lr_scheduler  # 学习率变化机制，这里需要根据不同机制修改不同代码
        self.evaluate = evaluate  # 评估指标
        self.n_gpu = n_gpu  # gpu个数，列表形式
        self.avg_batch_loss = avg_batch_loss  # 建议大的batch_size使用loss avg
        self.id_to_label = {value: key for key, value in label_to_id.items()}
        self.bioes = bioes
        self.sep = sep
        self.num_classes = num_classes
        self._reset()

    def _reset(self):

        self.train_entity_score = Entity_Score(id_to_label=self.id_to_label, logger=self.logger, bioes=self.bioes,
                                               sep=self.sep)
        self.val_entity_score = Entity_Score(id_to_label=self.id_to_label, logger=self.logger, bioes=self.bioes,
                                             sep=self.sep)
        self.train_evaluate = F1_score(num_classes=self.num_classes)
        self.val_evaluate = F1_score(num_classes=self.num_classes)
        self.batch_num = len(self.train_data)
        self.progressbar = ProgressBar(n_batch=self.batch_num, eval_name='acc', loss_name='loss')
        self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model, logger=self.logger)
        self.start_epoch = 1
        # 重载模型，进行训练
        if self.resume:
            arch = self.model_checkpoint.arch
            resume_path = os.path.join(self.model_checkpoint.checkpoint_dir.format(arch=arch),
                                       self.model_checkpoint.best_model_name.format(arch=arch))
            self.logger.info("\nLoading checkpoint: {} ...".format(resume_path))
            resume_list = restore_checkpoint(resume_path=resume_path, model=self.model, optimizer=self.optimizer)
            self.model = resume_list[0]
            self.optimizer = resume_list[1]
            best = resume_list[2]
            self.start_epoch = resume_list[3]

            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info("\nCheckpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # for p in model_parameters:
        #     print(p.size())
        params = sum([np.prod(p.size()) for p in model_parameters])
        # 总的模型参数量
        self.logger.info('trainable parameters: {:4}M'.format(params / 1000 / 1000))
        # 模型结构
        self.logger.info(self.model)

    # 保存模型信息
    def _save_info(self, epoch, val_loss):
        state = {
            'epoch': epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_loss': round(val_loss, 4)
        }
        return state

    # val数据集预测
    def _valid_epoch(self):
        self.model.eval()

        train_losses = AverageMeter()
        self.train_entity_score._reset()
        self.train_evaluate._reset()

        val_losses = AverageMeter()
        self.val_entity_score._reset()
        self.evaluate._reset()

        with torch.no_grad():
            for batch_idx, (inputs, target, length) in enumerate(self.train_data):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                length = length.to(self.device)
                batch_size = inputs.size(0)
                outputs = self.model(inputs, length)
                mask, target = batchify_with_label(inputs=inputs, target=target, outputs=outputs)
                loss = self.model.crf.neg_log_likelihood_loss(outputs, mask, target)
                if self.avg_batch_loss:
                    loss /= batch_size
                _, predicts = self.model.crf(outputs, mask)
                self.train_evaluate.update(predicts, target=target)
                train_losses.update(loss)
                if self.device != 'cpu':
                    predicts = predicts.cpu().numpy()
                    target = target.cpu().numpy()
                self.train_entity_score.update(pred_paths=predicts, label_paths=target, length=length)
            print('\n' + '-' * 15 + 'Train_Score' + '-' * 15)
            train_token_acc, train_token_f1 = self.train_evaluate.result()
            train_f1 = self.train_entity_score.result()

            for batch_idx, (inputs, target, length) in enumerate(self.val_data):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                length = length.to(self.device)
                batch_size = inputs.size(0)

                outputs = self.model(inputs, length)
                mask, target = batchify_with_label(inputs=inputs, target=target, outputs=outputs)
                loss = self.model.crf.neg_log_likelihood_loss(outputs, mask, target)
                if self.avg_batch_loss:
                    loss /= batch_size
                _, predicts = self.model.crf(outputs, mask)
                self.evaluate.update(predicts, target=target)

                val_losses.update(loss.item(), batch_size)
                if self.device != 'cpu':
                    predicts = predicts.cpu().numpy()
                    target = target.cpu().numpy()
                self.val_entity_score.update(pred_paths=predicts, label_paths=target, length=length)

            print('-' * 15 + 'Val_Score' + '-' * 15)
            val_token_acc, val_token_f1 = self.evaluate.result()
            val_f1 = self.val_entity_score.result()
            print('-' * 30)
        return {'val_loss': val_losses.avg,
                'val_token_acc': val_token_acc,
                'val_token_f1': val_token_f1,
                'val_f1': val_f1}, \
               {'train_token_acc': train_token_acc,
                'token_f1': train_token_f1,
                'train_f1': train_f1}

    # epoch训练
    def _train_epoch(self, mixup=False):
        self.model.train()
        train_loss = AverageMeter()
        self.batch_evaluate = F1_score(len(self.id_to_label) + 1)
        self.train_entity_score._reset()
        self.evaluate._reset()
        for batch_idx, train_data in enumerate(self.train_data):
            start = time.time()
            inputs, target, length = train_data
            inputs = inputs.to(self.device)
            target = target.to(self.device)
            length = length.to(self.device)
            batch_size = inputs.size(0)

            outputs = self.model(inputs, length)
            mask, target = batchify_with_label(inputs=inputs, target=target, outputs=outputs)
            loss = self.model.crf.neg_log_likelihood_loss(outputs, mask, target)
            if self.avg_batch_loss:
                loss /= batch_size

            train_loss.update(loss.item(), batch_size)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.verbose >= 1:
                self.progressbar.step(batch_idx=batch_idx,
                                      loss=loss.item(),
                                      use_time=time.time() - start)
        return {'loss': train_loss.avg, 'aug_loss': -1}

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            self.train_data = self.train_loader.make_iter()

            print("----------------- training start -----------------------")
            print("Epoch {i}/{epochs}......".format(i=epoch, epochs=self.start_epoch + self.epochs - 1))
            train_log = self._train_epoch()

            val_log, train_eval_log = self._valid_epoch()

            logs = dict(train_log, **val_log, **train_eval_log)
            self.logger.info(
                '\nEpoch: %d - '
                'loss: %.4f acc: %.4f - token_f1: %.4f - f1: %.4f'
                ' val_loss: %.4f - val_token_acc: %.4f - val_token_f1: %.4f - val_f1: %.4f' % (
                    epoch,
                    logs['loss'], logs['train_token_acc'], logs['token_f1'], logs['train_f1'],
                    logs['val_loss'], logs['val_token_acc'], logs['val_token_f1'], logs['val_f1'])
            )

            if self.lr_scheduler:
                self.lr_scheduler.step(logs['val_token_f1'], epoch)

            if self.training_monitor:
                self.training_monitor.step(logs)

            if self.model_checkpoint:
                state = self._save_info(epoch, val_loss=logs['val_loss'])
                self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor], state=state)

            if self.writer:
                self.writer.set_step(epoch, 'train')
                self.writer.add_scalar('loss', logs['loss'])
                self.writer.add_scalar('acc', logs['acc'])
                self.writer.set_step(epoch, 'valid')
                self.writer.add_scalar('val_loss', logs['val_loss'])
                self.writer.add_scalar('val_acc', logs['val_token_acc'])

            if self.early_stopping:
                self.early_stopping.step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break