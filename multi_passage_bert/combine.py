import os
import math
import json
import codecs
import logging
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from elasticsearch import Elasticsearch
import torch
from transformers import BertTokenizer

from utils.dataloader import TOKEN_CLS, TOKEN_SEP
from utils import model_utils, evaluator


def dbc_to_sbc(ustring):
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


def retrieve(es, es_index, es_doc_type, page_size, match_name, question):
    if match_name == 'chars':
        text = question['text']
        text = [tok for tok in text]
    elif match_name == 'bigrams':
        text = question['text']
        text = [text[tok_idx:tok_idx + 2] for tok_idx in range(len(text) - 1)]
    elif match_name == 'words':
        text = question['words']
    text = ' '.join(text)

    result = es.search(index=es_index, doc_type=es_doc_type, size=page_size, body={
        'query': {
            'match': {
                match_name: text
            }
        }
    })

    hits = result['hits']['hits']
    return hits


def check_retrieve(doc_set, retrieve_docs):
    result = 0
    for doc in retrieve_docs:
        doc = doc['_source']
        if doc['id'] in doc_set:
            result = 1
            break
    return result


def get_segments(retrieve_docs):
    segments = []
    for doc in retrieve_docs:
        segment = doc['_source']['text']
        segments.append(segment)
    return segments


def get_bert_data(tokenizer, max_question_len, max_context_len, question, segments, use_cpu=False):
    contexts = []
    inp_segments = []
    segments_text = []

    question = dbc_to_sbc(question)
    question = [ch for ch in question]

    if max_question_len > 0:
        question = question[:max_question_len]
    question = [TOKEN_CLS] + question + [TOKEN_SEP]
    question_id = tokenizer.convert_tokens_to_ids(question)

    for segment in segments:
        segment = dbc_to_sbc(segment)
        segment = [ch for ch in segment]

        if max_context_len > 0:
            segment = segment[:(max_context_len - len(question))]
        segment_id = tokenizer.convert_tokens_to_ids(segment)

        context = question_id + segment_id
        segment = question_id + segment
        inp_segment = [1] * len(question_id) + [0] * len(segment_id)

        contexts.append(context)
        inp_segments.append(inp_segment)
        segments_text.append(segment)

    contexts = [torch.LongTensor(np.array(item)) for item in contexts]
    inp_segments = [torch.LongTensor(np.array(item)) for item in inp_segments]

    contexts = [item.cpu() if use_cpu else item.cuda() for item in contexts]
    inp_segments = [item.cpu() if use_cpu else item.cuda() for item in inp_segments]

    return contexts, inp_segments, segments_text


def softmax(score):
    row_max = np.max(score, axis=1).reshape(-1, 1)
    score -= row_max
    score_exp = np.exp(score)
    prob = score_exp / np.sum(score_exp, axis=1, keepdims=True)
    return prob


def score_normalization(starts_sco, ends_sco, share_norm=True):
    if share_norm:
        shape = np.shape(starts_sco)
        starts_sco = starts_sco.reshape((1, -1))
        ends_sco = ends_sco.reshape((1, -1))

        starts_sco = softmax(starts_sco)
        ends_sco = softmax(ends_sco)

        starts_sco = starts_sco.reshape((shape[0], -1))
        ends_sco = ends_sco.reshape((shape[0], -1))
    else:
        starts_sco = softmax(starts_sco)
        ends_sco = softmax(ends_sco)
    return starts_sco, ends_sco


def get_answer(ranks_sco, starts_sco, ends_sco, starts, ends, segments_text):
    scores = []
    anss_range = []
    for idx in range(len(ranks_sco)):
        rank = ranks_sco[idx]
        start = starts[idx]
        end = ends[idx]

        score = rank * starts_sco[idx][start] * ends_sco[idx][end]
        scores.append(score)
        anss_range.append([start, end])

    ans_idx = np.argmax(scores)
    ans_range = anss_range[ans_idx]
    segment = segments_text[ans_idx]

    if (np.min(ans_range) > 0) and (ans_range[1] >= ans_range[0]):
        ans_text = segment[ans_range[0]: ans_range[1] + 1]
        ans_text = [str(ch) for ch in ans_text]
        ans_text = ''.join(ans_text)
    else:
        ans_text = ''

    return ans_text


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    logging.getLogger('elasticsearch').setLevel(logging.ERROR)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    model_utils.setup_seed(0)

    output_path = '%s/%s/combine' % (args.output_path, args.task)
    if args.with_ranker:
        output_path += '_rank'
    if args.share_norm:
        output_path += '_sn'
    if args.with_negitive:
        output_path += '_neg'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info('Testing %s' % args.task)
    es = Elasticsearch([{'host': args.es_host, 'port': 9200}])
    es_index = args.task.replace('_', '-') + '-seg'

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading pretrained model')
    classifier_model_path = '%s/%s/classifier/best.pth' % (args.model_path, args.task)

    reader_model_path = '%s/%s' % (args.model_path, args.task)
    if args.with_negitive:
        reader_model_path += '/reader_neg/best.pth'
    else:
        reader_model_path += '/reader/best.pth'

    if args.with_ranker:
        classifier_model, _, _, _ = model_utils.load(classifier_model_path)
        classifier_model = classifier_model.cpu() if args.use_cpu else classifier_model.cuda()
        classifier_model.eval()

    reader_model, _, _, _ = model_utils.load(reader_model_path)
    reader_model = reader_model.cpu() if args.use_cpu else reader_model.cuda()
    reader_model.eval()

    retrieve_total_num = 0
    retrieve_success_num = 0
    gold_answers = []
    pred_answers = []
    with torch.no_grad():
        dataset_path = args.task.split('_')
        with codecs.open('%s/%s/%s/dev_seg.txt' % (args.data_path, dataset_path[0], dataset_path[1]),
                         'r', 'utf-8') as fin:
            for line in tqdm(fin):
                if args.debug:
                    if len(gold_answers) > 30:
                        break

                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)
                question = line['question']
                segments = line['segments']
                segments_id = set([seg['id'] for seg in segments if seg['is_selected']])
                answers = line['answers']

                retrieve_segs = retrieve(es, es_index, args.es_doc_type, args.retrieve_size, args.retrieve_token_type,
                                         question)
                retrieve_total_num += 1
                retrieve_success_num += check_retrieve(segments_id, retrieve_segs)

                segments = get_segments(retrieve_segs)
                contexts, inp_segments, segments_text = get_bert_data(tokenizer, args.max_question_len,
                                                                      args.max_context_len, question['text'], segments,
                                                                      args.use_cpu)

                ranks_sco = []
                starts_sco = []
                ends_sco = []
                starts = []
                ends = []
                batch_num = math.ceil(len(contexts) / args.batch_size)
                for batch_idx in range(batch_num):
                    batch_end = args.batch_size * (batch_idx + 1)
                    batch_end = batch_end if batch_end <= len(contexts) else len(contexts)
                    b_contexts = contexts[args.batch_size * batch_idx: batch_end]
                    n_inp_segments = inp_segments[args.batch_size * batch_idx: batch_end]

                    if args.with_ranker:
                        b_ranks_sco, _ = classifier_model(b_contexts, n_inp_segments)
                        b_ranks_sco = b_ranks_sco[:, 1]
                    else:
                        b_ranks_sco = np.ones(np.shape(b_contexts)[0])

                    b_starts_sco, b_ends_sco, b_starts, b_ends = reader_model(b_contexts, n_inp_segments)
                    b_starts_sco = b_starts_sco.cpu().numpy()
                    b_ends_sco = b_ends_sco.cpu().numpy()
                    b_starts = b_starts.cpu().numpy()
                    b_ends = b_ends.cpu().numpy()

                    b_starts_sco, b_ends_sco = score_normalization(b_starts_sco, b_ends_sco, share_norm=args.share_norm)

                    ranks_sco.extend(list(b_ranks_sco))
                    starts_sco.extend(list(b_starts_sco))
                    ends_sco.extend(list(b_ends_sco))
                    starts.extend(list(b_starts))
                    ends.extend(list(b_ends))

                pred_ans = get_answer(ranks_sco, starts_sco, ends_sco, starts, ends, segments_text)

                pred_answers.append(pred_ans)
                gold_answers.append(answers)

    retrieve_recall = retrieve_success_num / retrieve_total_num
    reader_f1, reader_em, _, _ = evaluator.evaluate(gold_answers, pred_answers)

    with codecs.open('%s/log.txt' % output_path, 'w', 'utf-8') as fout:
        fout.write('retrieve_token_type: %s\n' % args.retrieve_token_type)
        fout.write('share_norm: %s\n' % args.share_norm)
        fout.write('with_ranker: %s\n' % args.with_ranker)
        fout.write('with_negitive: %s\n' % args.with_negitive)
        fout.write('retriever_recall: %.3f\n' % retrieve_recall)
        fout.write('reader_f1: %.3f\n' % reader_f1)
        fout.write('reader_em: %.3f\n' % reader_em)

    logging.info('retriever recall: %.3f, reader f1: %.3f, reader em: %.3f' % (retrieve_recall, reader_f1, reader_em))
    logging.info('complete testing')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--es_host', type=str,
                        default='10.79.169.35')
    parser.add_argument('--es_doc_type', type=str,
                        default='seg')

    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/multi_passage_bert/')
    parser.add_argument('--model_path', type=str,
                        default='../runtime/multi_passage_bert/')
    parser.add_argument('--task', type=str, choices=['dureader_search', 'dureader_zhidao', 'dureader_all'],
                        default='dureader_search')

    parser.add_argument('--retrieve_token_type', type=str, choices=['chars', 'bigrams', 'words'],
                        default='bigrams')
    parser.add_argument('--retrieve_size', type=int,
                        default=20)

    parser.add_argument('--pretrained_bert_path', dest='pretrained_bert_path',
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=35)

    parser.add_argument('--share_norm', type=bool,
                        default=True)
    parser.add_argument('--with_negitive', type=bool,
                        default=False)
    parser.add_argument('--with_ranker', type=bool,
                        default=True)

    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
