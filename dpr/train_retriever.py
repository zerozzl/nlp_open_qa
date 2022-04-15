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

from dpr.model import Retriever
from utils.dataloader import TOKEN_CLS, TOKEN_SEP
from utils import model_utils


class DprDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_question_len=0, max_context_len=0, debug=False):
        super(DprDataset, self).__init__()
        self.data = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = json.loads(line)

                question_id = line['question']['id']
                question = self.dbc_to_sbc(line['question']['text'])
                question = [ch for ch in question]
                if max_question_len > 0:
                    question = question[:max_question_len]

                question = [TOKEN_CLS] + question
                question = tokenizer.convert_tokens_to_ids(question)
                que_segment = [0] * len(question)

                segments_pos = []
                segments_neg = []
                for segment in line['segments']:
                    title = self.dbc_to_sbc(segment['document']['title'])
                    title = [ch for ch in title]

                    segment_text = self.dbc_to_sbc(segment['text'])
                    segment_text = [ch for ch in segment_text]

                    selected = segment['is_selected']

                    context = [TOKEN_CLS] + title + [TOKEN_SEP] + segment_text
                    ctx_segment = [0] * len(title) + [0] * 2 + [1] * len(segment_text)
                    if max_context_len > 0:
                        context = context[:max_context_len]
                        ctx_segment = ctx_segment[:max_context_len]
                    context = tokenizer.convert_tokens_to_ids(context)

                    if selected:
                        segments_pos.append([context, ctx_segment])
                    else:
                        segments_neg.append([context, ctx_segment])

                for pos_idx in range(len(segments_pos)):
                    for neg_idx in range(len(segments_neg)):
                        segments = []
                        segments.append([segments_pos[pos_idx][0], segments_pos[pos_idx][1], 1])
                        segments.append([segments_neg[neg_idx][0], segments_neg[neg_idx][1], 0])

                        self.data.append([question_id, question, que_segment, segments])

                if debug and (len(self.data) >= 100):
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


class Logger:
    def __init__(self, data_path=''):
        self.data_path = data_path

        if self.data_path != '':
            with codecs.open('%s/log.txt' % self.data_path, 'w', 'utf-8') as fout:
                fout.write('time\tepoch\tloss\taccuracy\tremark\n')

    def get_timestamp(self, format='%Y-%m-%d %H:%M:%S'):
        return datetime.strftime(datetime.now(), format)

    def write(self, epoch, loss, accuracy, remark=''):
        with codecs.open('%s/log.txt' % self.data_path, 'a', 'utf-8') as fout:
            fout.write('%s\t%s\t%.5f\t%.3f\t%s\n' % (
                self.get_timestamp(), epoch, loss, accuracy, remark))

    def draw_plot(self, data_path=''):
        if data_path == '':
            data_path = self.data_path

        eppch = []
        loss = []
        accuracy = []
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

        x_locator = MultipleLocator(int(len(eppch) / 5))
        y_locator = MultipleLocator(int(len(eppch) / 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        ax = plt.subplot2grid((1, 2), (0, 0), title='loss')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, loss)

        ax = plt.subplot2grid((1, 2), (0, 1), title='accuracy')
        ax.xaxis.set_major_locator(x_locator)
        ax.yaxis.set_major_locator(y_locator)
        ax.plot(eppch, accuracy)

        plt.rcParams['savefig.dpi'] = 200
        plt.savefig('%s/plot.jpg' % data_path)


def get_dataset(args, tokenizer):
    dataset_path = args.task.split('_')
    train_dataset = DprDataset('%s/%s/%s/train_dpr.txt' % (args.data_path, dataset_path[0], dataset_path[1]),
                               tokenizer, args.max_question_len, args.max_context_len, debug=args.debug)
    test_dataset = DprDataset('%s/%s/%s/dev_dpr.txt' % (args.data_path, dataset_path[0], dataset_path[1]),
                              tokenizer, args.max_question_len, args.max_context_len, debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    exists_ids = set()

    questions = []
    questions_seg = []
    contexts = []
    contexts_seg = []
    tags = []
    for record in data:
        if record[0] in exists_ids:
            continue

        exists_ids.add(record[0])

        questions.append(torch.LongTensor(np.array(record[1])))
        questions_seg.append(torch.LongTensor(np.array(record[2])))

        for seg in record[3]:
            contexts.append(torch.LongTensor(np.array(seg[0])))
            contexts_seg.append(torch.LongTensor(np.array(seg[1])))

    tags = [idx * 2 for idx in range(len(questions))]
    tags = torch.LongTensor(tags)

    return questions, questions_seg, contexts, contexts_seg, tags


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()

        questions, questions_seg, contexts, contexts_seg, tags = data

        questions = [item.cpu() if args.use_cpu else item.cuda() for item in questions]
        questions_seg = [item.cpu() if args.use_cpu else item.cuda() for item in questions_seg]
        contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
        contexts_seg = [item.cpu() if args.use_cpu else item.cuda() for item in contexts_seg]
        tags = tags.cpu() if args.use_cpu else tags.cuda()

        loss = model(questions, questions_seg, contexts, contexts_seg, decode=False, tags=tags)
        loss = loss.mean()

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, model):
    model.eval()

    if args.multi_gpu:
        model = model.module

    total_num = 0
    correct_num = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            questions, questions_seg, contexts, contexts_seg, tags = data

            questions = [item.cpu() if args.use_cpu else item.cuda() for item in questions]
            questions_seg = [item.cpu() if args.use_cpu else item.cuda() for item in questions_seg]
            contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
            contexts_seg = [item.cpu() if args.use_cpu else item.cuda() for item in contexts_seg]

            _, preds = model(questions, questions_seg, contexts, contexts_seg)

            tags = tags.cpu().numpy()
            preds = preds.cpu().numpy()

            total_num += len(preds)
            correct_num += (preds == tags).sum()

    accuracy = float(correct_num) / total_num
    return accuracy


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    if args.debug:
        args.batch_size = 3

    if args.multi_gpu:
        logging.info('run on multi GPU')
        torch.distributed.init_process_group(backend='nccl')

    model_utils.setup_seed(0)

    output_path = '%s/%s/retriever' % (args.output_path, args.task)
    if args.bert_freeze:
        output_path += '_freeze'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_logger = Logger(data_path=output_path)

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading dataset')
    train_dataset, test_dataset = get_dataset(args, tokenizer)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset, shuffle=True))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset, shuffle=False))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)

    best_acc = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info('loading pretrained model')
        model, optimizer, epoch, best_f1 = model_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info('creating model')
        model = Retriever('%s/config.json' % args.pretrained_bert_path,
                          '%s/pytorch_model.bin' % args.pretrained_bert_path,
                          args.bert_freeze)
        model = model.cpu() if args.use_cpu else model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info('begin training')
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)
        test_acc = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test accuracy: %s' % (epoch, args.epoch_size, test_acc))
        model_utils.save(output_path, 'last.pth', model, optimizer, epoch, test_acc)

        remark = ''
        if test_acc > best_acc:
            best_acc = test_acc
            remark = 'best'
            model_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_acc)

        model_logger.write(epoch, train_loss, test_acc, remark)

    logging.info('complete training')
    model_logger.draw_plot()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', type=str,
                        choices=['dureader_search', 'dureader_zhidao', 'dureader_all'],
                        default='dureader_zhidao')
    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default='../runtime/dpr/')
    parser.add_argument('--bert_freeze', type=bool,
                        default=False)
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=12)
    parser.add_argument('--epoch_size', type=int,
                        default=100)
    parser.add_argument('--learning_rate', type=float,
                        default=1e-5)
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
