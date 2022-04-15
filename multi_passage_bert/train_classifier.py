import os
import json
import codecs
import logging
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from multi_passage_bert.model import Classifier
from utils.dataloader import TOKEN_CLS, TOKEN_SEP
from utils import model_utils


class Logger:
    def __init__(self, data_path=''):
        self.data_path = data_path

        if self.data_path != '':
            with codecs.open('%s/log.txt' % self.data_path, 'w', 'utf-8') as fout:
                fout.write('time\tepoch\tloss\taccuracy\tprecision\trecall\tf1\tremark\n')

    def get_timestamp(self, format='%Y-%m-%d %H:%M:%S'):
        return datetime.strftime(datetime.now(), format)

    def write(self, epoch, loss, accuracy, precision, recall, f1, remark=''):
        with codecs.open('%s/log.txt' % self.data_path, 'a', 'utf-8') as fout:
            fout.write('%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%s\n' % (
                self.get_timestamp(), epoch, loss, accuracy, precision, recall, f1, remark))

    def draw_plot(self, data_path=''):
        if data_path == '':
            data_path = self.data_path

        eppch = []
        loss = []
        accuracy = []
        precision = []
        recall = []
        f1 = []

        with codecs.open('%s/log.txt' % data_path, 'r', 'utf-8') as fin:
            _ = fin.readline()
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = line.split('\t')
                eppch.append(int(line[1]) - 1)
                loss.append(float(line[2]))
                accuracy.append(float(line[3]))
                precision.append(float(line[4]))
                recall.append(float(line[5]))
                f1.append(float(line[6]))

        x_locator = MultipleLocator(int(len(eppch) / 5))
        y_locator = MultipleLocator(int(len(eppch) / 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        ax = plt.subplot2grid((2, 3), (0, 0), title='loss')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, loss)

        ax = plt.subplot2grid((2, 3), (0, 1), title='accuracy')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, accuracy)

        ax = plt.subplot2grid((2, 3), (1, 0), title='precision')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, precision)

        ax = plt.subplot2grid((2, 3), (1, 1), title='recall')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, recall)

        ax = plt.subplot2grid((2, 3), (1, 2), title='f1')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, f1)

        plt.rcParams['savefig.dpi'] = 200
        plt.savefig('%s/plot.jpg' % data_path)


class ClassifyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_question_len=0, max_context_len=0, debug=False):
        super(ClassifyDataset, self).__init__()
        self.data = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = json.loads(line)

                question = self.dbc_to_sbc(line['question']['text'])
                question = [ch for ch in question]
                if max_question_len > 0:
                    question = question[:max_question_len]

                question = [TOKEN_CLS] + question + [TOKEN_SEP]
                que_segment = [1] * len(question)

                for segment in line['segments']:
                    context = self.dbc_to_sbc(segment['text'])
                    context = [ch for ch in context]

                    selected = segment['is_selected']
                    selected = int(selected)

                    if max_context_len > 0:
                        context = context[:(max_context_len - len(question))]
                        inp_segment = que_segment + [0] * len(context)

                    context = question + context
                    context = tokenizer.convert_tokens_to_ids(context)

                    self.data.append([context, inp_segment, selected])

                if debug:
                    if len(self.data) >= 100:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def dbc_to_sbc(self, ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring


def get_dataset(args, tokenizer):
    dataset_path = args.task.split('_')
    train_dataset = ClassifyDataset('%s/%s/%s/train_seg.txt' % (args.data_path, dataset_path[0], dataset_path[1]),
                                    tokenizer, args.max_question_len, args.max_context_len, debug=args.debug)
    test_dataset = ClassifyDataset('%s/%s/%s/dev_seg.txt' % (args.data_path, dataset_path[0], dataset_path[1]),
                                   tokenizer, args.max_question_len, args.max_context_len, debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    contexts = data[:, 0].tolist()
    contexts = [torch.LongTensor(np.array(item)) for item in contexts]

    segments = data[:, 1].tolist()
    segments = [torch.LongTensor(np.array(item)) for item in segments]

    tags = data[:, 2].tolist()
    tags = torch.LongTensor(np.array(tags))

    return contexts, segments, tags


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        contexts, segments, tags = data

        contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
        segments = [item.cpu() if args.use_cpu else item.cuda() for item in segments]
        tags = tags.cpu() if args.use_cpu else tags.cuda()

        loss = model(contexts, segments, decode=False, tags=tags)
        loss = loss.mean()

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, model):
    pred_answers = []
    gold_answers = []

    model.eval()

    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            contexts, segments, tags = data

            contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
            segments = [item.cpu() if args.use_cpu else item.cuda() for item in segments]

            _, preds = model(contexts, segments)
            tags = tags.cpu().numpy()

            pred_answers.extend(preds)
            gold_answers.extend(tags)

    acc, pre, rec, f1 = calc_measure(gold_answers, pred_answers)
    return acc, pre, rec, f1


def calc_measure(golden_lists, predict_lists):
    acc = accuracy_score(golden_lists, predict_lists)
    pre = precision_score(golden_lists, predict_lists, average='macro')
    rec = recall_score(golden_lists, predict_lists, average='macro')
    f1 = f1_score(golden_lists, predict_lists, average='macro')
    return acc, pre, rec, f1


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if args.debug:
        args.batch_size = 3

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    model_utils.setup_seed(0)

    output_path = '%s/%s/classifier' % (args.output_path, args.task)
    if args.bert_freeze:
        output_path += '_freeze'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = Logger(data_path=output_path)

    logging.info("loading embedding")
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info("loading dataset")
    train_dataset, test_dataset = get_dataset(args, tokenizer)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)

    best_f1 = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info("loading pretrained model")
        model, optimizer, epoch, best_f1 = model_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info("creating model")
        model = Classifier('%s/config.json' % args.pretrained_bert_path,
                           '%s/pytorch_model.bin' % args.pretrained_bert_path,
                           args.bert_freeze)
        model = model.cpu() if args.use_cpu else model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info("begin training")
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_acc, test_pre, test_rec, test_f1 = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test accuracy: %s, precision: %s, recall: %s, f1: %s' % (
            epoch, args.epoch_size, test_acc, test_pre, test_rec, test_f1))
        model_utils.save(output_path, 'last.pth', model, optimizer, epoch, test_f1)

        remark = ''
        if test_f1 > best_f1:
            best_f1 = test_f1
            remark = 'best'
            model_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

        model_logger.write(epoch, train_loss, test_acc, test_pre, test_rec, test_f1, remark)

    logging.info("complete training")
    model_logger.draw_plot()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', type=str,
                        choices=['dureader_search', 'dureader_zhidao', 'dureader_all'],
                        default='dureader_search')
    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../runtime/multi_passage_bert/')
    parser.add_argument('--bert_freeze', type=bool,
                        default=False)
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=32)
    parser.add_argument('--epoch_size', type=int,
                        default=10)
    parser.add_argument('--learning_rate', type=float,
                        default=5e-5)
    parser.add_argument('--lr_warmup_proportion', type=float,
                        default=0.1)
    parser.add_argument('--lr_decay_gamma', type=float,
                        default=0.9)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', type=bool,
                        help='run with: -m torch.distributed.launch',
                        default=False)
    parser.add_argument('--local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
