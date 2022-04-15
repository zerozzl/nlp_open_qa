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


def check_retrieve(doc_id, retrieve_docs):
    result = 0
    for doc in retrieve_docs:
        doc = doc['_source']
        if doc['id'] == doc_id:
            result = 1
            break
    return result


def get_paragraphs(retrieve_docs):
    paragraphs = []
    for doc in retrieve_docs:
        paras = doc['_source']['text']
        paragraphs.append(paras)
    return paragraphs


def get_reader_data(tokenizer, max_question_len, max_context_len, question, paragraphs, use_cpu=False):
    contexts = []
    segments = []
    paragraphs_text = []

    question = dbc_to_sbc(question)
    question = [ch for ch in question]

    if max_question_len > 0:
        question = question[:max_question_len]
    question = [TOKEN_CLS] + question + [TOKEN_SEP]
    question_id = tokenizer.convert_tokens_to_ids(question)

    for para in paragraphs:
        para = dbc_to_sbc(para)

        para = [ch for ch in para]

        if max_context_len > 0:
            para = para[:(max_context_len - len(question))]
        para_id = tokenizer.convert_tokens_to_ids(para)

        context = question_id + para_id
        para_text = question_id + para
        segment = [1] * len(question_id) + [0] * len(para_id)

        contexts.append(context)
        segments.append(segment)
        paragraphs_text.append(para_text)

    contexts = [torch.LongTensor(np.array(item)) for item in contexts]
    segments = [torch.LongTensor(np.array(item)) for item in segments]

    contexts = [item.cpu() if use_cpu else item.cuda() for item in contexts]
    segments = [item.cpu() if use_cpu else item.cuda() for item in segments]

    return contexts, segments, paragraphs_text


def get_answer(retrieve_scores, starts_sco, ends_sco, starts, ends, paras_text, score_factor):
    scores = []
    anss_range = []
    for idx in range(len(starts)):
        retrieve_score = retrieve_scores[idx]
        start = starts[idx]
        end = ends[idx]
        start_score = starts_sco[idx][start]
        end_score = ends_sco[idx][end]

        score = (1 - score_factor) * retrieve_score + score_factor * (start_score + end_score)
        scores.append(score)
        anss_range.append([start, end])

    ans_idx = np.argmax(scores)
    ans_range = anss_range[ans_idx]
    segment = paras_text[ans_idx]

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

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model_utils.setup_seed(0)

    output_path = '%s/%s/combine' % (args.output_path, args.task)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info('Testing %s' % args.task)
    es = Elasticsearch([{'host': args.es_host, 'port': 9200}])
    es_index = args.task.replace('_', '-') + '-para'

    logging.info("loading embedding")
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info("loading pretrained model")
    model_path = '%s/%s/reader/best.pth' % (args.model_path, args.task)
    model, _, _, _ = model_utils.load(model_path)
    model = model.cpu() if args.use_cpu else model.cuda()
    model.eval()

    retrieve_total_num = 0
    retrieve_success_num = 0
    gold_answers = []
    pred_answers = []
    with torch.no_grad():
        dataset_path = args.task.split('_')
        with codecs.open('%s/%s/%s/dev_para.txt' % (args.data_path, dataset_path[0], dataset_path[1]),
                         'r', 'utf-8') as fin:
            for line in tqdm(fin):
                if args.debug:
                    if len(gold_answers) > 30:
                        break

                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)
                paragraph_id = line['paragraph']['id']
                question = line['question']
                answers = line['answers']
                answers = [item['text'] for item in answers]

                retrieve_paras = retrieve(es, es_index, args.es_doc_type, args.retrieve_size, args.retrieve_token_type,
                                          question)
                retrieve_scores = [item['_score'] for item in retrieve_paras]
                retrieve_total_num += 1
                retrieve_success_num += check_retrieve(paragraph_id, retrieve_paras)

                paragraphs = get_paragraphs(retrieve_paras)
                contexts, segments, paras_text = get_reader_data(tokenizer, args.max_question_len, args.max_context_len,
                                                                 question['text'], paragraphs, args.use_cpu)

                starts_sco = []
                ends_sco = []
                starts = []
                ends = []
                batch_num = math.ceil(len(contexts) / args.batch_size)
                for batch_idx in range(batch_num):
                    batch_end = args.batch_size * (batch_idx + 1)
                    batch_end = batch_end if batch_end <= len(contexts) else len(contexts)
                    b_contexts = contexts[args.batch_size * batch_idx: batch_end]
                    b_segments = segments[args.batch_size * batch_idx: batch_end]

                    b_starts_sco, b_ends_sco, b_starts, b_ends = model(b_contexts, b_segments)
                    b_starts_sco = b_starts_sco.cpu().numpy()
                    b_ends_sco = b_ends_sco.cpu().numpy()
                    b_starts = b_starts.cpu().numpy()
                    b_ends = b_ends.cpu().numpy()

                    starts_sco.extend(list(b_starts_sco))
                    ends_sco.extend(list(b_ends_sco))
                    starts.extend(list(b_starts))
                    ends.extend(list(b_ends))

                pred_ans = get_answer(retrieve_scores, starts_sco, ends_sco, starts, ends, paras_text,
                                      args.score_factor)

                pred_answers.append(pred_ans)
                gold_answers.append(answers)

    retrieve_recall = retrieve_success_num / retrieve_total_num
    reader_f1, reader_em, _, _ = evaluator.evaluate(gold_answers, pred_answers)

    with codecs.open('%s/log.txt' % output_path, 'w', 'utf-8') as fout:
        fout.write('retrieve_token_type: %s\n' % args.retrieve_token_type)
        fout.write('score_factor: %s\n' % args.score_factor)
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
                        default='para')

    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/bertserini/')
    parser.add_argument('--model_path', type=str,
                        default='../runtime/bertserini/')
    parser.add_argument('--task', type=str, choices=['dureader_search', 'dureader_zhidao', 'dureader_all'],
                        default='dureader_search')

    parser.add_argument('--retrieve_token_type', type=str, choices=['chars', 'bigrams', 'words'],
                        default='bigrams')
    parser.add_argument('--retrieve_size', type=int,
                        default=100)

    parser.add_argument('--pretrained_bert_path', dest='pretrained_bert_path',
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=35)
    parser.add_argument('--score_factor', type=int,
                        default=0.8)

    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
