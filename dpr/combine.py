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
import faiss
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


def load_segments(data_path, dataset_path):
    all_segments = {}
    doc_path = '%s/%s/%s/doc_seg.txt' % (data_path, dataset_path[0], dataset_path[1])
    with codecs.open(doc_path, 'r', 'utf-8') as fin:
        for line in tqdm(fin):
            line = line.strip()
            if line == '':
                continue

            line = json.loads(line)
            segments = line['segments']
            for segment in segments:
                seg_id = segment['id']
                seg_text = dbc_to_sbc(segment['text'])
                all_segments[seg_id] = seg_text
    return all_segments


def load_faiss_id(data_path):
    idx_to_id = {}
    with codecs.open(data_path, 'r', 'utf-8') as fin:
        for line in tqdm(fin):
            line = line.strip()
            if line == '':
                continue

            idx_to_id[len(idx_to_id)] = line
    return idx_to_id


def retrieve_from_faiss(args, tokenizer, model, faiss_idx_to_id, faiss_index, question):
    question = dbc_to_sbc(question['text'])
    question = [ch for ch in question]
    if args.max_question_len > 0:
        question = question[:args.max_question_len]

    question = [TOKEN_CLS] + question
    question = tokenizer.convert_tokens_to_ids(question)
    question_segment = [0] * len(question)

    question = torch.LongTensor(question).unsqueeze(0)
    question_segment = torch.LongTensor(question_segment).unsqueeze(0)
    question = question.cpu() if args.use_cpu else question.cuda()
    question_segment = question_segment.cpu() if args.use_cpu else question_segment.cuda()

    embedding = model.get_question_embedding(question, question_segment)
    embedding = embedding.detach().cpu().numpy()

    faiss_search_size = args.retrieve_size
    if args.use_es:
        faiss_search_size -= 1
    _, faiss_idx = faiss_index.search(embedding, faiss_search_size)

    search_ids = [faiss_idx_to_id[idx] for idx in faiss_idx[0]]
    return search_ids


def retrieve_from_es(es, es_index, es_doc_type, page_size, match_name, question):
    if args.es_token_type == 'chars':
        text = question['text']
        text = [tok for tok in text]
    elif args.es_token_type == 'bigrams':
        text = question['text']
        text = [text[tok_idx:tok_idx + 2] for tok_idx in range(len(text) - 1)]
    elif args.es_token_type == 'words':
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
    search_id = hits[0]['_source']['id']
    return search_id


def check_retrieve(segments_id, search_ids):
    result = 0
    for seg_id in search_ids:
        if seg_id in segments_id:
            result = 1
            break
    return result


def get_segments(all_segments, search_ids):
    segments = []
    for seg_id in search_ids:
        if seg_id in all_segments:
            segments.append(all_segments[seg_id])
    return segments


def get_reader_data(args, tokenizer, question, segments):
    contexts = []
    ctx_segments = []
    segments_text = []

    question = dbc_to_sbc(question['text'])
    question = [ch for ch in question]

    if args.max_question_len > 0:
        question = question[:args.max_question_len]
    question = [TOKEN_CLS] + question + [TOKEN_SEP]
    question_id = tokenizer.convert_tokens_to_ids(question)

    for segment in segments:
        segment = dbc_to_sbc(segment)
        segment = [ch for ch in segment]

        if args.max_context_len > 0:
            segment = segment[:(args.max_context_len - len(question))]
        segment_id = tokenizer.convert_tokens_to_ids(segment)

        context = question_id + segment_id
        segment = question + segment
        ctx_segment = [1] * len(question_id) + [0] * len(segment_id)

        contexts.append(context)
        ctx_segments.append(ctx_segment)
        segments_text.append(segment)

    contexts = [torch.LongTensor(np.array(item)) for item in contexts]
    ctx_segments = [torch.LongTensor(np.array(item)) for item in ctx_segments]

    contexts = [item.cpu() if args.use_cpu else item.cuda() for item in contexts]
    ctx_segments = [item.cpu() if args.use_cpu else item.cuda() for item in ctx_segments]

    return contexts, ctx_segments, segments_text


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

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    model_utils.setup_seed(0)

    logging.info('Testing %s' % args.task)

    output_path = '%s/%s/combine' % (args.output_path, args.task)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info('loading data')
    dataset_path = args.task.split('_')
    all_segments = load_segments(args.data_path, dataset_path)

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading pretrained model')
    retriever_model_path = '%s/%s/retriever/last.pth' % (args.model_path, args.task)
    retriever_model, _, _, _ = model_utils.load(retriever_model_path)
    retriever_model = retriever_model.cpu() if args.use_cpu else retriever_model.cuda()
    retriever_model.eval()

    reader_model_path = '%s/%s/reader/best.pth' % (args.model_path, args.task)
    reader_model, _, _, _ = model_utils.load(reader_model_path)
    reader_model = reader_model.cpu() if args.use_cpu else reader_model.cuda()
    reader_model.eval()

    classifier_model_path = '%s/%s/classifier/best.pth' % (args.model_path, args.task)
    classifier_model, _, _, _ = model_utils.load(classifier_model_path)
    classifier_model = classifier_model.cpu() if args.use_cpu else classifier_model.cuda()
    classifier_model.eval()

    logging.info('loading faiss index')
    faiss_id_path = '%s/%s/retriever/faiss_last.id' % (args.model_path, args.task)
    faiss_data_path = '%s/%s/retriever/faiss_last.data' % (args.model_path, args.task)
    faiss_idx_to_id = load_faiss_id(faiss_id_path)
    faiss_index = faiss.read_index(faiss_data_path)
    gpu_res = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss_index)

    es = None
    es_index = ''
    if args.use_es:
        es = Elasticsearch([{'host': args.es_host, 'port': 9200}])
        es_index = args.task.replace('_', '-') + '-seg'
        logging.info('Retrieving from elasticsearch index %s with %s' % (es_index, args.es_token_type))

    retrieve_total_num = 0
    retrieve_success_num = 0
    gold_answers = []
    pred_answers = []
    with torch.no_grad():
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

                search_ids = retrieve_from_faiss(args, tokenizer, retriever_model,
                                                 faiss_idx_to_id, faiss_index, question)
                if args.use_es:
                    es_id = retrieve_from_es(es, es_index, args.es_doc_type, 1, args.es_token_type, question)
                    search_ids.append(es_id)

                retrieve_total_num += 1
                retrieve_success_num += check_retrieve(segments_id, search_ids)

                segments = get_segments(all_segments, search_ids)
                contexts, ctx_segments, segments_text = get_reader_data(args, tokenizer, question, segments)

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
                    b_ctx_segments = ctx_segments[args.batch_size * batch_idx: batch_end]

                    b_ranks_sco, _ = classifier_model(b_contexts, b_ctx_segments)
                    b_ranks_sco = b_ranks_sco[:, 1]

                    b_starts_sco, b_ends_sco, b_starts, b_ends = reader_model(b_contexts, b_ctx_segments)
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
        fout.write('use_es: %s\n' % args.use_es)
        fout.write('es_token_type: %s\n' % args.es_token_type)
        fout.write('share_norm: %s\n' % args.share_norm)
        fout.write('retriever_recall: %.3f\n' % retrieve_recall)
        fout.write('reader_f1: %.3f\n' % reader_f1)
        fout.write('reader_em: %.3f\n' % reader_em)

    logging.info('retriever recall: %.3f, reader f1: %.3f, reader em: %.3f' % (retrieve_recall, reader_f1, reader_em))
    logging.info('complete testing')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', type=str,
                        choices=['dureader_search', 'dureader_zhidao', 'dureader_all'],
                        default='dureader_zhidao')
    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--pretrained_bert_path', type=str,
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--model_path', type=str,
                        default='../runtime/dpr/')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/dpr/')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--retrieve_size', type=int,
                        default=10)
    parser.add_argument('--batch_size', type=int,
                        default=35)
    parser.add_argument('--share_norm', type=bool,
                        default=True)
    parser.add_argument('--use_es', type=bool,
                        default=True)
    parser.add_argument('--es_host', type=str,
                        default='10.79.169.35')
    parser.add_argument('--es_doc_type', type=str,
                        default='seg')
    parser.add_argument('--es_token_type', type=str, choices=['chars', 'bigrams', 'words'],
                        default='bigrams')
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
