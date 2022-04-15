import os
import json
import codecs
import logging
import jieba
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from elasticsearch import Elasticsearch
import torch

from utils.dataloader import TOKEN_EDGES_START, TOKEN_EDGES_END, Tokenizer, load_pretrain_embedding
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
        paras = doc['_source']['paragraphs']
        for para in paras:
            paragraphs.append(para['text'])
    return paragraphs


def get_reader_data(token_type, use_bigram, max_question_len, max_context_len, tokenizer, bigram_tokenizer,
                    question, paragraphs, use_cpu=False):
    question_in = []
    question_bigram_in = []
    paragraphs_in = []
    paragraphs_bigram_in = []
    ext_match_in = []
    paragraphs_text = []

    question = dbc_to_sbc(question)
    if token_type == 'char':
        question = [ch for ch in question]
    elif token_type == 'word':
        question = list(jieba.cut(question))

    if max_question_len > 0:
        question = question[:max_question_len]
    question_id = tokenizer.convert_tokens_to_ids(question)

    question_bigram_id = []
    if use_bigram:
        question_bigram = [TOKEN_EDGES_START] + question + [TOKEN_EDGES_END]
        question_bigram = [[question_bigram[i - 1] + question_bigram[i]] + [
            question_bigram[i] + question_bigram[i + 1]] for i in range(1, len(question_bigram) - 1)]
        question_bigram_id = bigram_tokenizer.convert_tokens_to_ids(question_bigram)

    for para in paragraphs:
        para = dbc_to_sbc(para)

        if token_type == 'char':
            para = [ch for ch in para]
        elif token_type == 'word':
            para = list(jieba.cut(para))

        if max_context_len > 0:
            para = para[:max_context_len]
        para_id = tokenizer.convert_tokens_to_ids(para)

        ext_match = [int(tok in question) for tok in para]

        para_bigram_id = []
        if use_bigram:
            para_bigram = [TOKEN_EDGES_START] + para + [TOKEN_EDGES_END]
            para_bigram = [[para_bigram[i - 1] + para_bigram[i]] + [
                para_bigram[i] + para_bigram[i + 1]] for i in range(1, len(para_bigram) - 1)]
            para_bigram_id = bigram_tokenizer.convert_tokens_to_ids(para_bigram)

        question_in.append(question_id)
        question_bigram_in.append(question_bigram_id)
        paragraphs_in.append(para_id)
        paragraphs_bigram_in.append(para_bigram_id)
        ext_match_in.append(ext_match)
        paragraphs_text.append(para)

    question_in = [torch.LongTensor(np.array(item)) for item in question_in]
    question_bigram_in = [torch.LongTensor(np.array(item)) for item in question_bigram_in]
    paragraphs_in = [torch.LongTensor(np.array(item)) for item in paragraphs_in]
    paragraphs_bigram_in = [torch.LongTensor(np.array(item)) for item in paragraphs_bigram_in]
    ext_match_in = [torch.LongTensor(np.array(item)) for item in ext_match_in]

    question_in = [item.cpu() if use_cpu else item.cuda() for item in question_in]
    question_bigram_in = [item.cpu() if use_cpu else item.cuda() for item in question_bigram_in]
    paragraphs_in = [item.cpu() if use_cpu else item.cuda() for item in paragraphs_in]
    paragraphs_bigram_in = [item.cpu() if use_cpu else item.cuda() for item in paragraphs_bigram_in]
    ext_match_in = [item.cpu() if use_cpu else item.cuda() for item in ext_match_in]

    return question_in, question_bigram_in, paragraphs_in, paragraphs_bigram_in, ext_match_in, paragraphs_text


def get_reader_answer(starts, ends, starts_sco, ends_sco, paras_text):
    scores = []
    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        start_sco = starts_sco[i]
        end_sco = ends_sco[i]

        if end >= start:
            scores.append(start_sco[start] * end_sco[end])
        else:
            scores.append(0)

    ans_idx = np.argmax(scores)
    ans_start = starts[ans_idx]
    ans_end = ends[ans_idx]
    para = paras_text[ans_idx]
    para = para[ans_start: ans_end + 1]
    para = ''.join(para)
    return para


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
    es_index = args.task.replace('_', '-') + '-doc'

    logging.info("loading embedding")
    token_to_id, _ = load_pretrain_embedding(args.pretrained_emb_path,
                                             has_meta=True if (args.reader_token_type == 'word') else False,
                                             add_pad=True, add_unk=True, debug=args.debug)
    tokenizer = Tokenizer(token_to_id)

    bigram_tokenizer = None
    if args.use_bigram:
        bigram_to_id, pretrain_bigram_embed = load_pretrain_embedding(args.pretrained_bigram_emb_path,
                                                                      add_pad=True, add_unk=True, debug=args.debug)
        bigram_tokenizer = Tokenizer(bigram_to_id)

    logging.info("loading pretrained model")
    model, _, _, _ = model_utils.load(args.model_path)
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
                doc_id = line['document_id']
                question = line['question']
                answers = line['answers']
                answers = [item['text'] for item in answers]

                retrieve_docs = retrieve(es, es_index, args.es_doc_type, args.retrieve_size, args.retrieve_token_type,
                                         question)
                retrieve_total_num += 1
                retrieve_success_num += check_retrieve(doc_id, retrieve_docs)

                paragraphs = get_paragraphs(retrieve_docs)
                paragraphs.sort(key=lambda para: len(para), reverse=True)
                ques_in, ques_bigram_in, paras_in, paras_bigram_in, ext_match_in, paras_text = get_reader_data(
                    args.reader_token_type, args.use_bigram, args.max_question_len, args.max_context_len,
                    tokenizer, bigram_tokenizer, question['text'], paragraphs, args.use_cpu)

                starts_sco, ends_sco, starts, ends = model(ques_in, paras_in, ext_match_in,
                                                           question_bigram=ques_bigram_in,
                                                           ans_ctx_bigram=paras_bigram_in)
                starts_sco = starts_sco.cpu().numpy()
                ends_sco = ends_sco.cpu().numpy()
                starts = starts.cpu().numpy()
                ends = ends.cpu().numpy()

                pred_ans = get_reader_answer(starts, ends, starts_sco, ends_sco, paras_text)

                pred_answers.append(pred_ans)
                gold_answers.append(answers)

    retrieve_recall = retrieve_success_num / retrieve_total_num
    reader_f1, reader_em, _, _ = evaluator.evaluate(gold_answers, pred_answers)

    with codecs.open('%s/log.txt' % output_path, 'w', 'utf-8') as fout:
        fout.write('retrieve_token_type: %s\n' % args.retrieve_token_type)
        fout.write('reader_token_type: %s\n' % args.reader_token_type)
        fout.write('use_bigram: %s\n' % args.use_bigram)
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
                        default='doc')

    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/drqa/')
    parser.add_argument('--task', type=str, choices=['dureader_search', 'dureader_zhidao', 'dureader_all'],
                        default='dureader_search')
    parser.add_argument('--model_path', type=str,
                        default='../runtime/drqa/dureader_search/reader_char_rand/best.pth')

    parser.add_argument('--retrieve_token_type', type=str, choices=['chars', 'bigrams', 'words'],
                        default='words')
    parser.add_argument('--retrieve_size', type=int,
                        default=5)

    parser.add_argument('--reader_token_type', type=str, choices=['char', 'word'],
                        default='char')
    parser.add_argument('--use_bigram', type=bool,
                        default=False)
    parser.add_argument('--pretrained_emb_path', type=str,
                        help='[gigaword_chn.all.a2b.uni.ite50.vec],[news_tensite.pku.words.w2v50]',
                        default='../data/embeddings/gigaword_chn.all.a2b.uni.ite50.vec')
    parser.add_argument('--pretrained_bigram_emb_path', type=str,
                        default='../data/embeddings/gigaword_chn.all.a2b.bi.ite50.vec')
    parser.add_argument('--max_question_len', type=int,
                        help='64 for dureader',
                        default=64)
    parser.add_argument('--max_context_len', type=int,
                        default=1024)

    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
