import os
import json
import codecs
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from elasticsearch import Elasticsearch


def retrieve(es_host, es_index, es_doc_type, page_size, match_name, match_value):
    es = Elasticsearch([{'host': es_host, 'port': 9200}])

    result = es.search(index=es_index, doc_type=es_doc_type, size=page_size, body={
        'query': {
            'match': {
                match_name: match_value
            }
        }
    })

    hits = result['hits']['hits']
    return hits


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    logging.getLogger('elasticsearch').setLevel(logging.ERROR)

    logging.info('Retrieving %s with %s' % (args.task, args.token_type))
    es_index = args.task.replace('_', '-') + '-doc'

    output_path = '%s/%s/retriever_%s' % (args.output_path, args.task, args.token_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset_path = args.task.split('_')
    with codecs.open('%s/log.txt' % output_path, 'w', 'utf-8') as fout:
        data_splits = ['train', 'dev']
        for split in data_splits:
            total_num = 0
            success_num = 0
            with codecs.open('%s/%s/%s/%s_para.txt' % (args.data_path, dataset_path[0], dataset_path[1], split),
                             'r', 'utf-8') as fin:
                for line in tqdm(fin):
                    line = line.strip()
                    if line == '':
                        continue

                    line = json.loads(line)
                    doc_id = line['document_id']
                    question = line['question']

                    if args.token_type == 'chars':
                        text = question['text']
                        text = [tok for tok in text]
                    elif args.token_type == 'bigrams':
                        text = question['text']
                        text = [text[tok_idx:tok_idx + 2] for tok_idx in range(len(text) - 1)]
                    elif args.token_type == 'words':
                        text = question['words']
                    text = ' '.join(text)

                    success = False
                    docs = retrieve(args.es_host, es_index, args.es_doc_type, args.retrieve_size, args.token_type, text)
                    for doc in docs:
                        doc = doc['_source']
                        if doc['id'] == doc_id:
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

    parser.add_argument('--es_host', type=str,
                        default='10.79.169.35')
    parser.add_argument('--es_doc_type', type=str,
                        default='doc')
    parser.add_argument('--task', type=str, choices=['dureader_search', 'dureader_zhidao', 'dureader_all'],
                        default='dureader_search')
    parser.add_argument('--data_path', type=str,
                        default='../data/datasets/')
    parser.add_argument('--output_path', type=str,
                        default='../runtime/drqa/')
    parser.add_argument('--token_type', type=str, choices=['chars', 'bigrams', 'words'],
                        default='words')
    parser.add_argument('--retrieve_size', type=int,
                        default=5)
    parser.add_argument('--debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
