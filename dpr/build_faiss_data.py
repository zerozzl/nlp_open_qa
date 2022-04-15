import os
import json
import codecs
import logging
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import BertTokenizer

import torch
import faiss

from utils.dataloader import TOKEN_CLS, TOKEN_SEP
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


def build_embedding(args):
    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    model_path = '%s/%s/retriever/last.pth' % (args.model_path, args.task)
    model, _, _, _ = model_utils.load(model_path)
    model = model.cpu() if args.use_cpu else model.cuda()
    model.eval()

    dataset_path = args.task.split('_')
    doc_path = '%s/%s/%s/doc_seg.txt' % (args.data_path, dataset_path[0], dataset_path[1])
    logging.info('Reading %s' % doc_path)

    output_path = '%s/%s/retriever/last.emb' % (args.output_path, args.task)
    logging.info('Writing to %s' % output_path)

    processed_num = 0
    with torch.no_grad():
        with codecs.open(doc_path, 'r', 'utf-8') as fin, codecs.open(output_path, 'w', 'utf-8') as fout:
            for line in tqdm(fin):
                processed_num += 1
                if args.debug:
                    if processed_num >= 100:
                        break

                line = line.strip()
                if line == '':
                    continue

                line = json.loads(line)
                title = dbc_to_sbc(line['title'])
                title = [ch for ch in title]

                segments = line['segments']
                for segment in segments:
                    seg_id = segment['id']
                    seg_text = dbc_to_sbc(segment['text'])
                    seg_text = [ch for ch in seg_text]

                    context = [TOKEN_CLS] + title + [TOKEN_SEP] + seg_text
                    context_segment = [0] * len(title) + [0] * 2 + [1] * len(seg_text)
                    if args.max_context_len > 0:
                        context = context[:args.max_context_len]
                        context_segment = context_segment[:args.max_context_len]
                    context = tokenizer.convert_tokens_to_ids(context)

                    context = torch.LongTensor(context).unsqueeze(0)
                    context_segment = torch.LongTensor(context_segment).unsqueeze(0)
                    context = context.cpu() if args.use_cpu else context.cuda()
                    context_segment = context_segment.cpu() if args.use_cpu else context_segment.cuda()

                    embedding = model.get_context_embedding(context, context_segment)
                    embedding = embedding.squeeze(0).detach().cpu().numpy()

                    embedding = [str(emb) for emb in embedding]
                    embedding = ','.join(embedding)

                    fout.write('%s\t%s\n' % (seg_id, embedding))

    logging.info('Complete build embedding')


def build_faiss_index(args):
    ids = []
    embed_path = '%s/%s/retriever/last.emb' % (args.output_path, args.task)
    logging.info('Reading %s' % embed_path)
    with codecs.open(embed_path, 'r', 'utf-8') as fin:
        batch_data = []

        line = fin.readline()
        id, embed = line.split('\t')
        embed = [float(e) for e in embed.split(',')]

        ids.append(id)
        batch_data.append(embed)
        embed_size = len(embed)

        faiss_index = faiss.IndexFlatIP(embed_size)
        for line in tqdm(fin):
            id, embed = line.split('\t')
            embed = [float(e) for e in embed.split(',')]

            ids.append(id)
            batch_data.append(embed)

            if len(batch_data) % args.batch_size == 0:
                batch_data = np.array(batch_data).astype('float32')
                faiss_index.add(batch_data)
                batch_data = []

            if args.debug and (faiss_index.ntotal >= 2000):
                break

        if len(batch_data) > 0:
            batch_data = np.array(batch_data).astype('float32')
            faiss_index.add(batch_data)

        assert len(ids) == faiss_index.ntotal
        logging.info('Total index num %s' % faiss_index.ntotal)

    faiss_id_path = '%s/%s/retriever/faiss_last.id' % (args.output_path, args.task)
    logging.info('Writing faiss id to %s' % faiss_id_path)
    with codecs.open(faiss_id_path, 'w', 'utf-8') as fout:
        for id in ids:
            fout.write('%s\n' % id)

    faiss_data_path = '%s/%s/retriever/faiss_last.data' % (args.output_path, args.task)
    logging.info('Writing faiss data to %s' % faiss_data_path)
    faiss.write_index(faiss_index, faiss_data_path)


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    logging.info('Begin build embedding')
    build_embedding(args)

    logging.info('Begin build faiss index')
    build_faiss_index(args)


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
    parser.add_argument('--max_context_len', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=4096)
    parser.add_argument('--use_cpu', type=bool,
                        default=False)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
