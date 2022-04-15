import os
import json
import codecs
import logging
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from transformers import BertTokenizer

from multi_passage_bert.model import Reader
from utils.dataloader import TOKEN_CLS, TOKEN_SEP
from utils.logger import Logger
from utils import model_utils, evaluator


class MrDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_question_len=0, max_context_len=0,
                 with_negitive=True, do_train=False, debug=False):
        super(MrDataset, self).__init__()
        self.data = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = json.loads(line)

                question = self.dbc_to_sbc(line['question']['text'])
                question = [ch for ch in question]
                if max_question_len > 0:
                    question = question[:max_question_len]

                question = [TOKEN_CLS] + question + [TOKEN_SEP]
                question = tokenizer.convert_tokens_to_ids(question)
                que_segment = [1] * len(question)

                for segment in line['segments']:
                    context = self.dbc_to_sbc(segment['text'])
                    context = [ch for ch in context]

                    selected = segment['is_selected']
                    if (not with_negitive) and (not selected):
                        continue

                    selected = int(selected)

                    answer = segment['answer']
                    ans_text = self.dbc_to_sbc(answer['text'])
                    ans_start = answer['pos_idx']
                    if do_train:
                        if selected:
                            if ans_text == '':
                                continue

                            ans_ctx = [context[:ans_start], context[ans_start:ans_start + len(ans_text)],
                                       context[ans_start + len(ans_text):]]
                            ans_text = [ans_text]

                            ans_ctx = [[ch for ch in piece] for piece in ans_ctx]

                            ans_tags = [len(ans_ctx[0]), len(ans_ctx[0]) + (len(ans_ctx[1]) - 1)]
                            if (ans_tags[0] < 0) or (ans_tags[1] < 0) or (ans_tags[0] > ans_tags[1]):
                                continue
                            ans_ctx = ans_ctx[0] + ans_ctx[1] + ans_ctx[2]

                            if max_context_len > 0:
                                if ((ans_tags[0] + len(question)) >= max_context_len) or \
                                        ((ans_tags[1] + len(question)) >= max_context_len):
                                    continue
                                ans_ctx = ans_ctx[:(max_context_len - len(question))]
                                inp_segment = que_segment + [0] * len(ans_ctx)
                        else:
                            ans_ctx = context
                            ans_text = [ans_text]
                            ans_tags = [-1, -1]

                            if max_context_len > 0:
                                ans_ctx = ans_ctx[:(max_context_len - len(question))]
                                inp_segment = que_segment + [0] * len(ans_ctx)
                    else:
                        ans_ctx = context
                        ans_text = [ans_text]
                        ans_tags = [-1, -1]

                        if max_context_len > 0:
                            ans_ctx = ans_ctx[:(max_context_len - len(question))]
                            inp_segment = que_segment + [0] * len(ans_ctx)

                    ans_ctx_text = ans_ctx
                    ans_ctx = tokenizer.convert_tokens_to_ids(ans_ctx)

                    self.data.append([question, ans_ctx, inp_segment, ans_tags, ans_ctx_text, ans_text])

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
    train_dataset = MrDataset('%s/%s/%s/train_seg.txt' % (args.data_path, dataset_path[0], dataset_path[1]),
                              tokenizer, args.max_question_len, args.max_context_len,
                              with_negitive=args.with_negitive, do_train=True, debug=args.debug)
    test_dataset = MrDataset('%s/%s/%s/dev_seg.txt' % (args.data_path, dataset_path[0], dataset_path[1]),
                             tokenizer, args.max_question_len, args.max_context_len,
                             with_negitive=args.with_negitive, do_train=False, debug=args.debug)

    return train_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)

    contexts = []
    segments = []
    ans_tags = []
    contexts_text = []
    ans_texts = []
    for item in data:
        question = item[0]
        ans_ctx = item[1]
        segment = item[2]
        ans_tag = item[3]
        ctx_text = item[4]
        ans_text = item[5]

        contexts.append(question + ans_ctx)
        segments.append(segment)
        if min(ans_tag) >= 0:
            ans_tags.append([tag + len(question) for tag in ans_tag])
        else:
            ans_tags.append([0, 0])
        contexts_text.append(question + ctx_text)
        ans_texts.append(ans_text)

    contexts = [torch.LongTensor(np.array(s)) for s in contexts]
    ans_tags = torch.LongTensor(ans_tags)
    segments = [torch.LongTensor(np.array(s)) for s in segments]

    return contexts, segments, ans_tags, contexts_text, ans_texts


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        contexts, segments, ans_tags, _, _ = data

        contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
        segments = [item.cpu() if args.use_cpu else item.cuda() for item in segments]
        ans_tags = ans_tags.cpu() if args.use_cpu else ans_tags.cuda()

        loss = model(contexts, segments, decode=False, tags=ans_tags)
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
            contexts, segments, _, contexts_text, answers = data

            contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
            segments = [item.cpu() if args.use_cpu else item.cuda() for item in segments]

            _, _, starts, ends = model(contexts, segments)
            starts = starts.cpu().numpy()
            ends = ends.cpu().numpy()

            for i in range(len(starts)):
                ctx = contexts_text[i]
                start = starts[i]
                end = ends[i]
                answer = answers[i]
                if '' in answer:
                    continue

                if (start > 0) and (end > 0) and (start <= end):
                    pred_answers.append(''.join([str(ch) for ch in ctx[start:end + 1]]))
                else:
                    pred_answers.append('')

                gold_answers.append(answer)

    f1_score, em_score, _, _ = evaluator.evaluate(gold_answers, pred_answers)
    return f1_score, em_score


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    if args.debug:
        args.batch_size = 3

    if args.multi_gpu:
        logging.info('run on multi GPU')
        torch.distributed.init_process_group(backend='nccl')

    model_utils.setup_seed(0)

    output_path = '%s/%s/reader' % (args.output_path, args.task)
    if args.with_negitive:
        output_path += '_neg'
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

    best_f1 = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info('loading pretrained model')
        model, optimizer, epoch, best_f1 = model_utils.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info('creating model')
        model = Reader('%s/config.json' % args.pretrained_bert_path,
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
        test_f1, test_em = evaluate(args, test_dataloader, model)

        logging.info('epoch[%s/%s], train loss: %s' % (epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], test f1: %s, em: %s' % (epoch, args.epoch_size, test_f1, test_em))
        model_utils.save(output_path, 'last.pth', model, optimizer, epoch, test_f1)

        remark = ''
        if test_f1 > best_f1:
            best_f1 = test_f1
            remark = 'best'
            model_utils.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

        model_logger.write(epoch, train_loss, test_f1, test_em, remark)

    logging.info('complete training')
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
    parser.add_argument('--with_negitive', type=bool,
                        default=False)
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
