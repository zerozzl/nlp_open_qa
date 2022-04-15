import os
import json
import codecs
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import BertTokenizer
from elasticsearch import Elasticsearch

import torch
import faiss

from utils.dataloader import TOKEN_CLS
from utils import model_utils


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


def load_faiss_id(data_path):
    idx_to_id = {}
    with codecs.open(data_path, 'r', 'utf-8') as fin:
        for line in fin:
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


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    logging.getLogger('elasticsearch').setLevel(logging.ERROR)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    output_path = '%s/%s/retriever_faiss' % (args.output_path, args.task)
    if args.use_es:
        output_path += '_es_%s' % args.es_token_type
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info('loading embedding')
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info('loading pretrained model')
    model_path = '%s/%s/retriever/last.pth' % (args.model_path, args.task)
    model, _, _, _ = model_utils.load(model_path)
    model = model.cpu() if args.use_cpu else model.cuda()
    model.eval()

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

    with torch.no_grad():
        dataset_path = args.task.split('_')
        with codecs.open('%s/log.txt' % output_path, 'w', 'utf-8') as fout:
            data_splits = ['train', 'dev']
            for split in data_splits:
                total_num = 0
                success_num = 0
                data_split_path = '%s/%s/%s/%s_dpr.txt' % (args.data_path, dataset_path[0], dataset_path[1], split)

                logging.info('Reading: %s' % data_split_path)
                with codecs.open(data_split_path, 'r', 'utf-8') as fin:
                    for line in tqdm(fin):
                        if args.debug:
                            if total_num >= 100:
                                break
                        line = line.strip()
                        if line == '':
                            continue

                        line = json.loads(line)
                        question = line['question']
                        segments = line['segments']
                        segments_id = set([seg['id'] for seg in segments if seg['is_selected']])

                        search_ids = retrieve_from_faiss(args, tokenizer, model, faiss_idx_to_id, faiss_index, question)
                        if args.use_es:
                            es_id = retrieve_from_es(es, es_index, args.es_doc_type, 1, args.es_token_type, question)
                            search_ids.append(es_id)

                        success = False
                        for seg_id in search_ids:
                            if seg_id in segments_id:
                                success = True
                                break

                        total_num += 1
                        if success:
                            success_num += 1

                recall = success_num / total_num
                logging.info('%s data recall %.3f' % (split, recall))
                fout.write('%s data recall %.3f\n' % (split, recall))


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
    parser.add_argument('--retrieve_size', type=int,
                        default=100)
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
