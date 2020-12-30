# encoding:utf-8
import torch
from tqdm import tqdm
from .predict_utils import get_entity
from ..train.train_utils import restore_checkpoint, model_device
from ..train.trainer import batchify_with_label
from ..utils.utils import AverageMeter
from ..train.metrics import Entity_Score
import json
from ..io import data_transformer


# 单个模型进行预测
class Predicter(object):
    def __init__(self,
                 model,
                 test_data,
                 logger,
                 label_to_id,
                 checkpoint_path,
                 evaluate,
                 total_evaluate,
                 n_gpu=0,
                 bioes=False,
                 sep='_',
                 i2v=None,
                 i2l=None):
        self.model = model
        self.test_data = test_data
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.n_gpu = n_gpu
        self.id_to_label = {value: tag for tag, value in label_to_id.items()}
        self.avg_batch_loss = True
        self.evaluate = evaluate  # 评估指标
        self.bioes = bioes
        self.total_evaluate = total_evaluate
        self.sep = sep
        self.i2v = i2v
        self.i2l = i2l

        self._reset()

    # 重载模型
    def _reset(self):
        self.batch_num = len(self.test_data)
        self.test_entity_score = Entity_Score(id_to_label=self.id_to_label, logger=self.logger, bioes=self.bioes,
                                              sep=self.sep)

        self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model, logger=self.logger)
        if self.checkpoint_path:
            self.logger.info("\nLoading checkpoint: {} ...".format(self.checkpoint_path))
            resume_list = restore_checkpoint(resume_path=self.checkpoint_path, model=self.model)
            self.model = resume_list[0]
            self.logger.info("\nCheckpoint '{}' loaded".format(self.checkpoint_path))

    # batch预测
    def _predict_batch(self, inputs, length):
        with torch.no_grad():
            outputs = self.model(inputs, length)
            mask, _ = batchify_with_label(inputs=inputs, outputs=outputs, is_train_mode=False)
            _, predicts = self.model.crf(outputs, mask)
            batch_result = []
            for index, (text, path) in enumerate(zip(inputs, predicts)):
                if self.device != 'cpu':
                    path = path.cpu().numpy()
                result = get_entity(path=path, tag_map=self.id_to_label, bioes=self.bioes)
                batch_result.append(result)
            return batch_result, predicts

    def _valid_epoch(self):
        self.model.eval()
        test_losses = AverageMeter()
        self.test_entity_score._reset()
        with torch.no_grad():
            for batch_idx, (inputs, target, length) in enumerate(self.test_data):
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                length = length.to(self.device)
                batch_size = inputs.size(0)

                outputs = self.model(inputs, length)
                mask, target = batchify_with_label(inputs=inputs, target=target, outputs=outputs, is_train_mode=True)
                loss = self.model.crf.neg_log_likelihood_loss(outputs, mask, target)
                if self.avg_batch_loss:
                    loss /= batch_size
                _, predicts = self.model.crf(outputs, mask)
                self.evaluate.update(predicts, target=target)
                self.total_evaluate.update(predicts, target=target)

                test_losses.update(loss.item(), batch_size)
                if self.device != 'cpu':
                    predicts = predicts.cpu().numpy()
                    target = target.cpu().numpy()
                else:
                    predicts = predicts.numpy()
                    target = target.numpy()
                self.test_entity_score.update(pred_paths=predicts, label_paths=target, length=length)
        test_acc, test_f1 = self.evaluate.result()
        total_metrics = self.total_evaluate.result()

        return {'test_loss': test_losses.avg,
                'test_acc': test_acc,
                'test_f1': test_f1
                }, total_metrics

    # 预测test数据集
    def predict(self):
        self.model.eval()
        metrics, total_metrics = self._valid_epoch()
        print('----------Test total score:')
        for tag, scores in total_metrics.items():
            if not tag.endswith('avg'):
                print('=' * 5 + self.id_to_label[int(tag)] + '=' * 5)
            else:
                print('=' * 5 + tag + '=' * 5)
            print(json.dumps(scores))

        print('----------Test entity score:-------')
        f = self.test_entity_score.result()
        predictions = []
        tags = []
        text_epoch = []
        for batch_idx, (inputs, targets, length) in tqdm(enumerate(self.test_data), total=self.batch_num,
                                                         desc='test_data'):
            inputs = inputs.to(self.device)
            length = length.to(self.device)
            y_pred_batch, preds = self._predict_batch(inputs=inputs, length=length)
            text_epoch.extend(
                [[self.i2v[x] for x in sentence[:l]] for sentence, l in zip(inputs.cpu().tolist(), length)])
            tags.extend([[self.i2l[x] for x in sentence[:l]] for sentence, l in zip(targets.cpu().tolist(), length)])
            predictions.extend(
                [[self.i2l[x] for x in sentence[:l]] for sentence, l in zip(preds.cpu().tolist(), length)])
        return metrics, text_epoch, tags, predictions, f
