import os
import json
import uuid
import codecs
import logging
import jieba
from tqdm import tqdm
from elasticsearch import Elasticsearch
from torch.utils.data import Dataset

TOKEN_PAD = '[PAD]'
TOKEN_UNK = '[UNK]'
TOKEN_CLS = '[CLS]'
TOKEN_SEP = '[SEP]'
TOKEN_EDGES_START = '<s>'
TOKEN_EDGES_END = '</s>'


class MrDataset(Dataset):
    def __init__(self, data_path, token_type, tokenizer, use_bigram=False, bigram_tokenizer=None,
                 max_context_len=0, max_question_len=0, do_pad=False, pad_token=TOKEN_PAD,
                 do_to_id=False, do_sort=False, do_train=False, for_bert=False, debug=False):
        super(MrDataset, self).__init__()
        self.data = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = json.loads(line)

                question = self.dbc_to_sbc(line['question']['text'])
                context = self.dbc_to_sbc(line['paragraph']['text'])
                answers = line['answers']

                if token_type == 'char':
                    question = [ch for ch in question]
                elif token_type == 'word':
                    question = list(jieba.cut(question))

                if max_question_len > 0:
                    question = question[:max_question_len]

                segment = []
                if for_bert:
                    question = [TOKEN_CLS] + question + [TOKEN_SEP]
                    segment = [1] * len(question)

                ans_tags = []
                if do_train:
                    ans_text = self.dbc_to_sbc(answers[0]['text'])
                    ans_start = answers[0]['pos_idx']
                    if ans_text == '':
                        continue

                    ans_ctx = [context[:ans_start], context[ans_start:ans_start + len(ans_text)],
                               context[ans_start + len(ans_text):]]
                    ans_text = [ans_text]

                    if token_type == 'char':
                        ans_ctx = [[ch for ch in piece] for piece in ans_ctx]
                    elif token_type == 'word':
                        ans_ctx = [list(jieba.cut(piece)) for piece in ans_ctx]

                    ans_tags = [len(ans_ctx[0]), len(ans_ctx[0]) + (len(ans_ctx[1]) - 1)]
                    if (ans_tags[0] < 0) or (ans_tags[1] < 0) or (ans_tags[0] > ans_tags[1]):
                        continue
                    ans_ctx = ans_ctx[0] + ans_ctx[1] + ans_ctx[2]

                    if max_context_len > 0:
                        if for_bert:
                            if ((ans_tags[0] + len(question)) >= max_context_len) or \
                                    ((ans_tags[1] + len(question)) >= max_context_len):
                                continue
                            ans_ctx = ans_ctx[:(max_context_len - len(question))]
                            segment = segment + [0] * len(ans_ctx)
                        else:
                            if (ans_tags[0] >= max_context_len) or (ans_tags[1] >= max_context_len):
                                continue
                            ans_ctx = ans_ctx[:max_context_len]
                else:
                    ans_text = [self.dbc_to_sbc(ans['text']) for ans in answers]

                    if token_type == 'char':
                        ans_ctx = [ch for ch in context]
                    elif token_type == 'word':
                        ans_ctx = list(jieba.cut(context))

                    if max_context_len > 0:
                        if for_bert:
                            ans_ctx = ans_ctx[:(max_context_len - len(question))]
                            segment = segment + [0] * len(ans_ctx)
                        else:
                            ans_ctx = ans_ctx[:max_context_len]

                ans_ctx_len = len(ans_ctx)

                ans_ctx_ext_match = [int(tok in question) for tok in ans_ctx]

                question_bigram = []
                ans_ctx_bigram = []
                if use_bigram:
                    question_bigram = [TOKEN_EDGES_START] + question + [TOKEN_EDGES_END]
                    question_bigram = [[question_bigram[i - 1] + question_bigram[i]] + [
                        question_bigram[i] + question_bigram[i + 1]] for i in
                                       range(1, len(question_bigram) - 1)]

                    ans_ctx_bigram = [TOKEN_EDGES_START] + ans_ctx + [TOKEN_EDGES_END]
                    ans_ctx_bigram = [[ans_ctx_bigram[i - 1] + ans_ctx_bigram[i]] + [
                        ans_ctx_bigram[i] + ans_ctx_bigram[i + 1]] for i in range(1, len(ans_ctx_bigram) - 1)]

                ans_ctx_text = ans_ctx
                if do_to_id:
                    question = tokenizer.convert_tokens_to_ids(question)
                    ans_ctx = tokenizer.convert_tokens_to_ids(ans_ctx)
                    if use_bigram:
                        question_bigram = bigram_tokenizer.convert_tokens_to_ids(question_bigram)
                        ans_ctx_bigram = bigram_tokenizer.convert_tokens_to_ids(ans_ctx_bigram)

                self.data.append([question, ans_ctx, ans_tags, ans_ctx_len, ans_ctx_text, ans_text,
                                  ans_ctx_ext_match, question_bigram, ans_ctx_bigram, segment])

                if debug:
                    if len(self.data) >= 10:
                        break

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

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


class DuReaderData:
    @staticmethod
    def transform_paragraphs(datapath, output_path):
        datasets = ['zhidao', 'search']
        data_splits = ['train', 'dev']
        for dataset in datasets:
            dataset_path = '%s/%s' % (output_path, dataset)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            doc_data = []
            for split in data_splits:
                mr_data = []
                src_file = '%s/%s.%s.json' % (datapath, dataset, split)

                logging.info('reading %s' % src_file)
                ques_count = 0
                ques_entity_count = 0
                with codecs.open(src_file, 'r', 'utf-8') as fin:
                    for line in fin:
                        try:
                            line = line.strip()
                            if line == '':
                                continue

                            ques_count += 1
                            line = json.loads(line)

                            docs = []
                            documents = line['documents']
                            for document in documents:
                                paras = []
                                for para_idx in range(len(document['paragraphs'])):
                                    paras.append({'id': str(uuid.uuid1()),
                                                  'idx': para_idx,
                                                  'text': document['paragraphs'][para_idx],
                                                  'words': document['segmented_paragraphs'][para_idx]})

                                doc = {'id': str(uuid.uuid1()),
                                       'is_selected': document['is_selected'],
                                       'title': document['title'],
                                       'most_related_para': document['most_related_para'],
                                       'paragraphs': paras}
                                docs.append(doc)
                                doc_data.append(doc)

                            if line['question_type'] != 'ENTITY':
                                continue

                            ques_entity_count += 1
                            question = {'id': str(uuid.uuid1()),
                                        'text': line['question'],
                                        'words': line['segmented_question']}

                            answers = []
                            for ans in line['entity_answers']:
                                if len(ans) > 0:
                                    answers.append(ans)
                            if len(answers) == 0:
                                continue

                            for doc in docs:
                                if not doc['is_selected']:
                                    continue
                                para = doc['paragraphs'][doc['most_related_para']]
                                for answer in answers:
                                    ans_ents = []
                                    for piece in answer:
                                        piece = piece.strip()
                                        if piece == '':
                                            continue
                                        answer_idx = para['text'].find(piece)
                                        if answer_idx >= 0:
                                            ans_ents.append([piece, answer_idx])
                                    if len(ans_ents) > 0:
                                        # ans_ents.sort(key=lambda item: item[1])
                                        if split == 'train':
                                            mr_data.append({'question': question,
                                                            'document_id': doc['id'],
                                                            'paragraph': para,
                                                            'answers': [
                                                                {'text': ans_ents[0][0], 'pos_idx': ans_ents[0][1]}
                                                            ]})
                                        else:
                                            mr_data.append({'question': question,
                                                            'document_id': doc['id'],
                                                            'paragraph': para,
                                                            'answers': [
                                                                {'text': item[0], 'pos_idx': -1} for item in ans_ents
                                                            ]})
                        except:
                            pass

                logging.info(
                    'mr question count %s, entity type %s, keep %s' % (ques_count, ques_entity_count, len(mr_data)))
                tgt_file = '%s/%s_para.txt' % (dataset_path, split)
                logging.info('saving %s' % tgt_file)
                with codecs.open(tgt_file, 'w', 'utf-8') as fout:
                    for line in mr_data:
                        fout.write('%s\n' % (json.dumps(line)))

            logging.info('document count %s' % len(doc_data))
            tgt_file = '%s/doc_para.txt' % dataset_path
            logging.info('saving %s' % tgt_file)
            with codecs.open(tgt_file, 'w', 'utf-8') as fout:
                for line in doc_data:
                    fout.write('%s\n' % (json.dumps(line)))

        logging.info('combine datasets')
        combine_path = '%s/all' % output_path
        if not os.path.exists(combine_path):
            os.makedirs(combine_path)
        for split in data_splits:
            mr_data = []
            for dataset in datasets:
                src_file = '%s/%s/%s_para.txt' % (output_path, dataset, split)
                with codecs.open(src_file, 'r', 'utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        mr_data.append(line)
            logging.info('mr data %s' % len(mr_data))

            tgt_file = '%s/%s_para.txt' % (combine_path, split)
            logging.info('saving %s' % tgt_file)
            with codecs.open(tgt_file, 'w', 'utf-8') as fout:
                for line in mr_data:
                    fout.write('%s\n' % line)

        doc_data = []
        for dataset in datasets:
            src_file = '%s/%s/doc_para.txt' % (output_path, dataset)
            with codecs.open(src_file, 'r', 'utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line == '':
                        continue

                    doc_data.append(line)
        logging.info('doc data %s' % len(doc_data))

        tgt_file = '%s/doc_para.txt' % combine_path
        logging.info('saving %s' % tgt_file)
        with codecs.open(tgt_file, 'w', 'utf-8') as fout:
            for line in doc_data:
                fout.write('%s\n' % line)

        logging.info('complete transform DuReader')

    @staticmethod
    def transform_segments(datapath, output_path, segment_size=100, sliding_size=50):
        datasets = ['zhidao', 'search']
        data_splits = ['train', 'dev']
        for dataset in datasets:
            dataset_path = '%s/%s' % (output_path, dataset)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            doc_data = []
            for split in data_splits:
                mr_data = []
                src_file = '%s/%s.%s.json' % (datapath, dataset, split)

                logging.info('reading %s' % src_file)
                ques_count = 0
                ques_entity_count = 0
                with codecs.open(src_file, 'r', 'utf-8') as fin:
                    for line in fin:
                        try:
                            line = line.strip()
                            if line == '':
                                continue

                            ques_count += 1
                            line = json.loads(line)

                            docs = []
                            documents = line['documents']
                            for document in documents:
                                most_related_para = document['most_related_para']
                                words = []
                                segments = []
                                paras_to_segments = []
                                for para_idx in range(len(document['paragraphs'])):
                                    para_segments = set()
                                    para_words = document['segmented_paragraphs'][para_idx]
                                    while len(para_words) > 0:
                                        para_segments.add(len(segments))
                                        piece_size = segment_size - len(words)
                                        if piece_size >= len(para_words):
                                            piece_size = len(para_words)

                                        words = words + para_words[:piece_size]
                                        para_words = para_words[piece_size:]

                                        if len(words) == segment_size:
                                            segments.append(words)
                                            words = words[sliding_size:]

                                    para_segments = list(para_segments)
                                    para_segments.sort()
                                    paras_to_segments.append(para_segments)

                                if len(words) > 0:
                                    segments.append(words)
                                    words = []

                                segments = [{'id': str(uuid.uuid1()), 'text': ''.join(seg), 'words': seg} \
                                            for seg in segments]
                                doc = {'id': str(uuid.uuid1()),
                                       'is_selected': document['is_selected'],
                                       'title': document['title'],
                                       'most_related_segments': paras_to_segments[most_related_para],
                                       'segments': segments}
                                docs.append(doc)
                                doc_data.append(doc)

                            if line['question_type'] != 'ENTITY':
                                continue

                            ques_entity_count += 1
                            question = {'id': str(uuid.uuid1()),
                                        'text': line['question'],
                                        'words': line['segmented_question']}

                            answers = []
                            for ans in line['entity_answers']:
                                if len(ans) > 0:
                                    answers.append(ans)
                            if len(answers) == 0:
                                continue

                            for doc in docs:
                                if not doc['is_selected']:
                                    continue
                                related_segments = [doc['segments'][seg] for seg in doc['most_related_segments']]
                                for segment in related_segments:
                                    for answer in answers:
                                        ans_ents = []
                                        for piece in answer:
                                            piece = piece.strip()
                                            if piece == '':
                                                continue
                                            answer_idx = segment['text'].find(piece)
                                            if answer_idx >= 0:
                                                ans_ents.append([piece, answer_idx])
                                        if len(ans_ents) > 0:
                                            # ans_ents.sort(key=lambda item: item[1])
                                            if split == 'train':
                                                mr_data.append({'question': question,
                                                                'document_id': doc['id'],
                                                                'segment': segment,
                                                                'answers': [
                                                                    {'text': ans_ents[0][0], 'pos_idx': ans_ents[0][1]}
                                                                ]})
                                            else:
                                                mr_data.append({'question': question,
                                                                'document_id': doc['id'],
                                                                'segment': segment,
                                                                'answers': [
                                                                    {'text': item[0], 'pos_idx': -1} for item in
                                                                    ans_ents
                                                                ]})
                        except:
                            pass

                logging.info(
                    'mr question count %s, entity type %s, keep %s' % (ques_count, ques_entity_count, len(mr_data)))
                tgt_file = '%s/%s_seg_gold.txt' % (dataset_path, split)
                logging.info('saving %s' % tgt_file)
                with codecs.open(tgt_file, 'w', 'utf-8') as fout:
                    for line in mr_data:
                        fout.write('%s\n' % (json.dumps(line)))

            logging.info('document count %s' % len(doc_data))
            tgt_file = '%s/doc_seg.txt' % dataset_path
            logging.info('saving %s' % tgt_file)
            with codecs.open(tgt_file, 'w', 'utf-8') as fout:
                for line in doc_data:
                    fout.write('%s\n' % (json.dumps(line)))

        logging.info('combine datasets')
        combine_path = '%s/all' % output_path
        if not os.path.exists(combine_path):
            os.makedirs(combine_path)
        for split in data_splits:
            mr_data = []
            for dataset in datasets:
                src_file = '%s/%s/%s_seg_gold.txt' % (output_path, dataset, split)
                with codecs.open(src_file, 'r', 'utf-8') as fin:
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            continue

                        mr_data.append(line)
            logging.info('mr data %s' % len(mr_data))

            tgt_file = '%s/%s_seg_gold.txt' % (combine_path, split)
            logging.info('saving %s' % tgt_file)
            with codecs.open(tgt_file, 'w', 'utf-8') as fout:
                for line in mr_data:
                    fout.write('%s\n' % line)

        doc_data = []
        for dataset in datasets:
            src_file = '%s/%s/doc_seg.txt' % (output_path, dataset)
            with codecs.open(src_file, 'r', 'utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line == '':
                        continue

                    doc_data.append(line)
        logging.info('doc data %s' % len(doc_data))

        tgt_file = '%s/doc_seg.txt' % combine_path
        logging.info('saving %s' % tgt_file)
        with codecs.open(tgt_file, 'w', 'utf-8') as fout:
            for line in doc_data:
                fout.write('%s\n' % line)

        logging.info('complete transform DuReader')

    @staticmethod
    def es_create_doc_index(es_host, es_index, es_doc_type):
        es = Elasticsearch([{'host': es_host, 'port': 9200}])

        mapping = {
            "mappings": {
                es_doc_type: {
                    "properties": {
                        "id": {
                            "type": "keyword"
                        },
                        "is_selected": {
                            "type": "boolean"
                        },
                        "title": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "chars": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "bigrams": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "words": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "most_related_para": {
                            "type": "long"
                        },
                        "paragraphs": {
                            "properties": {
                                "id": {
                                    "type": "keyword"
                                },
                                "idx": {
                                    "type": "long"
                                },
                                "text": {
                                    "type": "text",
                                    "analyzer": "ik_smart",
                                    "fields": {
                                        "cn": {
                                            "type": "text",
                                            "analyzer": "ik_smart"
                                        },
                                        "en": {
                                            "type": "text",
                                            "analyzer": "english"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if es.indices.exists(index=es_index):
            logging.info('index %s already exists' % es_index)
        else:
            result = es.indices.create(index=es_index, body=mapping)
            logging.info('create index %s' % es_index)
            logging.info(result)

    @staticmethod
    def es_save_doc(datapath, es_host, es_index, es_doc_type):
        DuReaderData.es_create_doc_index(es_host, es_index, es_doc_type)

        es = Elasticsearch([{'host': es_host, 'port': 9200}])
        logging.info('saving %s' % datapath)
        with codecs.open('%s/doc.txt' % datapath, 'r', 'utf-8') as fin:
            for line in tqdm(fin):
                line = json.loads(line)

                chars = []
                bigrams = []
                words = []
                paras = []
                for para in line['paragraphs']:
                    text = para['text']
                    chars.extend([tok for tok in text])

                    bigrams.extend([text[tok_idx:tok_idx + 2] for tok_idx in range(len(text) - 1)])
                    words.extend([tok for tok in para['words']])
                    paras.append({
                        'id': para['id'],
                        'idx': para['idx'],
                        'text': para['text']
                    })

                chars = ' '.join(chars)
                bigrams = ' '.join(bigrams)
                words = ' '.join(words)

                doc = {
                    'id': line['id'],
                    'is_selected': line['is_selected'],
                    'title': line['title'],
                    'chars': chars,
                    'bigrams': bigrams,
                    'words': words,
                    'most_related_para': line['most_related_para'],
                    'paragraphs': paras
                }

                es.index(index=es_index, doc_type=es_doc_type, body=doc)

    @staticmethod
    def es_create_para_index(es_host, es_index, es_doc_type):
        es = Elasticsearch([{'host': es_host, 'port': 9200}])

        mapping = {
            "mappings": {
                es_doc_type: {
                    "properties": {
                        "id": {
                            "type": "keyword"
                        },
                        "text": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "chars": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "bigrams": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "words": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        }
                    }
                }
            }
        }

        if es.indices.exists(index=es_index):
            logging.info('index %s already exists' % es_index)
        else:
            result = es.indices.create(index=es_index, body=mapping)
            logging.info('create index %s' % es_index)
            logging.info(result)

    @staticmethod
    def es_save_paragraph(datapath, es_host, es_index, es_doc_type):
        DuReaderData.es_create_para_index(es_host, es_index, es_doc_type)

        es = Elasticsearch([{'host': es_host, 'port': 9200}])
        logging.info('saving %s' % datapath)
        with codecs.open('%s/doc.txt' % datapath, 'r', 'utf-8') as fin:
            for line in tqdm(fin):
                line = json.loads(line)
                for para in line['paragraphs']:
                    text = para['text']
                    chars = [tok for tok in text]
                    bigrams = [text[tok_idx:tok_idx + 2] for tok_idx in range(len(text) - 1)]
                    words = para['words']

                    chars = ' '.join(chars)
                    bigrams = ' '.join(bigrams)
                    words = ' '.join(words)

                    record = {
                        'id': para['id'],
                        'text': para['text'],
                        'chars': chars,
                        'bigrams': bigrams,
                        'words': words
                    }

                    es.index(index=es_index, doc_type=es_doc_type, body=record)

    @staticmethod
    def es_create_segment_index(es_host, es_index, es_doc_type):
        es = Elasticsearch([{'host': es_host, 'port': 9200}])

        mapping = {
            "mappings": {
                es_doc_type: {
                    "properties": {
                        "id": {
                            "type": "keyword"
                        },
                        "text": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "chars": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "bigrams": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        },
                        "words": {
                            "type": "text",
                            "analyzer": "ik_smart",
                            "fields": {
                                "cn": {
                                    "type": "text",
                                    "analyzer": "ik_smart"
                                },
                                "en": {
                                    "type": "text",
                                    "analyzer": "english"
                                }
                            }
                        }
                    }
                }
            }
        }

        if es.indices.exists(index=es_index):
            logging.info('index %s already exists' % es_index)
        else:
            result = es.indices.create(index=es_index, body=mapping)
            logging.info('create index %s' % es_index)
            logging.info(result)

    @staticmethod
    def es_save_segment(datapath, es_host, es_index, es_doc_type):
        DuReaderData.es_create_segment_index(es_host, es_index, es_doc_type)

        es = Elasticsearch([{'host': es_host, 'port': 9200}])
        logging.info('saving %s' % datapath)
        with codecs.open('%s/doc_seg.txt' % datapath, 'r', 'utf-8') as fin:
            for line in tqdm(fin):
                line = json.loads(line)
                for segment in line['segments']:
                    text = segment['text']
                    chars = [tok for tok in text]
                    bigrams = [text[tok_idx:tok_idx + 2] for tok_idx in range(len(text) - 1)]
                    words = segment['words']

                    chars = ' '.join(chars)
                    bigrams = ' '.join(bigrams)
                    words = ' '.join(words)

                    record = {
                        'id': segment['id'],
                        'text': segment['text'],
                        'chars': chars,
                        'bigrams': bigrams,
                        'words': words
                    }

                    es.index(index=es_index, doc_type=es_doc_type, body=record)

    @staticmethod
    def create_segments(datapath, es_host, retrive_token_type, debug=False):
        def retrieve(es_index, es_doc_type, page_size, match_name, match_value):
            result = es.search(index=es_index, doc_type=es_doc_type, size=page_size, body={
                'query': {
                    'match': {
                        match_name: match_value
                    }
                }
            })

            hits = result['hits']['hits']
            hits = [hit['_source'] for hit in hits]
            return hits

        es = Elasticsearch([{'host': es_host, 'port': 9200}])
        datasets = ['zhidao', 'search']
        data_split = ['train', 'dev']
        for dataset in datasets:
            for ds in data_split:
                logging.info('create %s %s segment data' % (dataset, ds))

                data = {}
                with codecs.open('%s/%s/%s_seg_gold.txt' % (datapath, dataset, ds), 'r', 'utf-8') as fin:
                    for line in tqdm(fin):
                        if debug:
                            if len(data) > 10:
                                break

                        line = line.strip()
                        if line == '':
                            continue

                        line = json.loads(line)
                        question_id = line['question']['id']

                        if question_id in data:
                            record = data[question_id]
                            record['segments'].append({
                                'id': line['segment']['id'],
                                'is_selected': True,
                                'text': line['segment']['text'],
                                'words': line['segment']['words'],
                                'answer': line['answers'][0]
                            })
                            record['answers'].extend([ans['text'] for ans in line['answers']])
                        else:
                            record = {
                                'question': line['question'],
                                'document_id': line['document_id'],
                                'segments': [
                                    {
                                        'id': line['segment']['id'],
                                        'is_selected': True,
                                        'text': line['segment']['text'],
                                        'words': line['segment']['words'],
                                        'answer': line['answers'][0]
                                    }
                                ],
                                'answers': [ans['text'] for ans in line['answers']]
                            }
                        record['answers'] = list(set(record['answers']))
                        data[question_id] = record

                es_index = 'dureader-%s-seg' % dataset
                es_doc_type = 'seg'
                for question_id in tqdm(data):
                    record = data[question_id]
                    segments_id = set([seg['id'] for seg in record['segments']])

                    if retrive_token_type == 'chars':
                        text = record['question']['text']
                        text = [tok for tok in text]
                    elif retrive_token_type == 'bigrams':
                        text = record['question']['text']
                        text = [text[tok_idx:tok_idx + 2] for tok_idx in range(len(text) - 1)]
                    elif retrive_token_type == 'words':
                        text = record['question']['words']
                    text = ' '.join(text)

                    ques_ret_docs = retrieve(es_index, es_doc_type, 10, retrive_token_type, text)
                    for ret_doc in ques_ret_docs:
                        if ret_doc['id'] in segments_id:
                            continue
                        record['segments'].append({
                            'id': ret_doc['id'],
                            'is_selected': False,
                            'text': ret_doc['text'],
                            'words': ret_doc['words'].split(),
                            'answer': {
                                'text': '',
                                'pos_idx': -1
                            }
                        })
                        segments_id.add(ret_doc['id'])

                    for answer in record['answers']:
                        if retrive_token_type == 'chars':
                            text = [tok for tok in answer]
                        elif retrive_token_type == 'bigrams':
                            text = [answer[tok_idx:tok_idx + 2] for tok_idx in range(len(answer) - 1)]
                        elif retrive_token_type == 'words':
                            list(jieba.cut(answer))
                        text = ' '.join(text)

                        ans_ret_docs = retrieve(es_index, es_doc_type, 10, retrive_token_type, text)
                        for ret_doc in ans_ret_docs:
                            if ret_doc['id'] in segments_id:
                                continue
                            record['segments'].append({
                                'id': ret_doc['id'],
                                'is_selected': False,
                                'text': ret_doc['text'],
                                'words': ret_doc['words'].split(),
                                'answer': {
                                    'text': '',
                                    'pos_idx': -1
                                }
                            })
                            segments_id.add(ret_doc['id'])

                    data[question_id] = record

                tgt_file = '%s/%s/%s_seg.txt' % (datapath, dataset, ds)
                logging.info('saving %s' % tgt_file)
                with codecs.open(tgt_file, 'w', 'utf-8') as fout:
                    for question_id in data:
                        record = data[question_id]
                        fout.write('%s\n' % (json.dumps(record)))

        logging.info('complete create DuReader segment data ')

    @staticmethod
    def create_dpr(data_path, debug=False):
        datasets = ['zhidao', 'search']
        data_splits = ['train', 'dev']
        for dataset in datasets:
            logging.info('processing dataset %s' % dataset)

            doc_dict = {}
            segment_to_doc = {}
            logging.info('reading doc')
            with codecs.open('%s/%s/doc_seg.txt' % (data_path, dataset), 'r', 'utf-8') as fin:
                for line in tqdm(fin):
                    line = line.strip()
                    if line == '':
                        continue

                    line = json.loads(line)
                    doc_dict[line['id']] = {
                        'id': line['id'],
                        'is_selected': line['is_selected'],
                        'title': line['title']
                    }

                    for segment in line['segments']:
                        segment_to_doc[segment['id']] = line['id']

                    if debug and (len(doc_dict) >= 1000):
                        break

            for split in data_splits:
                mr_data = []
                logging.info('reading %s data' % split)
                with codecs.open('%s/%s/%s_seg.txt' % (data_path, dataset, split), 'r', 'utf-8') as fin:
                    for line in tqdm(fin):
                        line = line.strip()
                        if line == '':
                            continue

                        line = json.loads(line)
                        segments = []
                        has_neg = False
                        for segment in line['segments']:
                            if segment['is_selected']:
                                segment['document'] = doc_dict[segment_to_doc[segment['id']]]
                                segments.append(segment)
                            elif not has_neg:
                                segment['document'] = doc_dict[segment_to_doc[segment['id']]]
                                segments.append(segment)
                                has_neg = True

                        mr_data.append({
                            'question': line['question'],
                            'document': doc_dict[line['document_id']],
                            'segments': segments,
                            'answers': line['answers']
                        })

                logging.info('saving %s data' % split)
                with codecs.open('%s/%s/%s_dpr.txt' % (data_path, dataset, split), 'w', 'utf-8') as fout:
                    for record in tqdm(mr_data):
                        fout.write('%s\n' % (json.dumps(record)))

        logging.info('complete create dpr data ')

    @staticmethod
    def statistics_paragraphs(datapath):
        context_len_dict = {128: 0, 256: 0, 384: 0, 400: 0, 512: 0, 768: 0, 1024: 0, 2048: 0, 4096: 0}
        question_len_dict = {16: 0, 24: 0, 32: 0, 64: 0, 128: 0}

        datasets = ['zhidao', 'search', 'all']
        data_granularity = ['char', 'word']
        data_split = ['train', 'dev']
        for dataset in datasets:
            logging.info('statistics %s' % dataset)
            for dg in data_granularity:
                for ds in data_split:
                    logging.info('%s level %s data count' % (dg, ds))
                    with codecs.open('%s/%s/%s_para.txt' % (datapath, dataset, ds), 'r', 'utf-8') as fin:
                        for line in fin:
                            line = json.loads(line)

                            if dg == 'char':
                                context_len = len(line['paragraph']['text'])
                                que_len = len(line['question']['text'])
                            elif dg == 'word':
                                context_len = len(line['paragraph']['words'])
                                que_len = len(line['question']['words'])

                            for cl in context_len_dict:
                                if context_len <= cl:
                                    context_len_dict[cl] = context_len_dict[cl] + 1
                                    break
                            for ql in question_len_dict:
                                if que_len <= ql:
                                    question_len_dict[ql] = question_len_dict[ql] + 1
                                    break

                    logging.info('context length: %s' % str(context_len_dict))
                    logging.info('question length: %s' % str(question_len_dict))

                    for cl in context_len_dict:
                        context_len_dict[cl] = 0
                    for ql in question_len_dict:
                        question_len_dict[ql] = 0

    @staticmethod
    def statistics_segments(datapath):
        context_len_dict = {128: 0, 256: 0, 384: 0, 400: 0, 512: 0, 768: 0, 1024: 0, 2048: 0, 4096: 0}
        question_len_dict = {16: 0, 24: 0, 32: 0, 64: 0, 128: 0}

        datasets = ['zhidao', 'search', 'all']
        data_granularity = ['char', 'word']
        data_split = ['train', 'dev']
        for dataset in datasets:
            logging.info('statistics %s' % dataset)
            for dg in data_granularity:
                for ds in data_split:
                    logging.info('%s level %s data count' % (dg, ds))
                    with codecs.open('%s/%s/%s_seg_gold.txt' % (datapath, dataset, ds), 'r', 'utf-8') as fin:
                        for line in fin:
                            line = json.loads(line)

                            if dg == 'char':
                                context_len = len(line['segment']['text'])
                                que_len = len(line['question']['text'])
                            elif dg == 'word':
                                context_len = len(line['segment']['words'])
                                que_len = len(line['question']['words'])

                            for cl in context_len_dict:
                                if context_len <= cl:
                                    context_len_dict[cl] = context_len_dict[cl] + 1
                                    break
                            for ql in question_len_dict:
                                if que_len <= ql:
                                    question_len_dict[ql] = question_len_dict[ql] + 1
                                    break

                    logging.info('context length: %s' % str(context_len_dict))
                    logging.info('question length: %s' % str(question_len_dict))

                    for cl in context_len_dict:
                        context_len_dict[cl] = 0
                    for ql in question_len_dict:
                        question_len_dict[ql] = 0


class Tokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    def convert_tokens_to_ids(self, tokens, unk_token=TOKEN_UNK):
        ids = []
        for token in tokens:
            if isinstance(token, str):
                ids.append(self.token_to_id.get(token, self.token_to_id[unk_token]))
            else:
                ids.append([self.token_to_id.get(t, self.token_to_id[unk_token]) for t in token])
        return ids

    def convert_ids_to_tokens(self, ids, max_sent_len=0):
        tokens = [self.id_to_token[i] for i in ids]
        if max_sent_len > 0:
            tokens = tokens[:max_sent_len]
        return tokens


def load_pretrain_embedding(filepath, has_meta=False, add_pad=False, pad_token=TOKEN_PAD,
                            add_unk=False, unk_token=TOKEN_UNK, debug=False):
    with codecs.open(filepath, 'r', 'utf-8', errors='ignore') as fin:
        token_to_id = {}
        embed = []

        if has_meta:
            meta_info = fin.readline().strip().split()

        first_line = fin.readline().strip().split()
        embed_size = len(first_line) - 1

        if add_pad:
            token_to_id[pad_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        if add_unk:
            token_to_id[unk_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        token_to_id[first_line[0]] = len(token_to_id)
        embed.append([float(x) for x in first_line[1:]])

        for line in fin:
            line = line.split()

            if len(line) != embed_size + 1:
                continue
            if line[0] in token_to_id:
                continue

            token_to_id[line[0]] = len(token_to_id)
            embed.append([float(x) for x in line[1:]])

            if debug:
                if len(embed) >= 1000:
                    break

    return token_to_id, embed


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    logging.getLogger('elasticsearch').setLevel(logging.ERROR)

    # DuReaderData.transform_paragraphs('/sdc/zerozzl/temp/DuReader/preprocessed', '/sdc/zerozzl/temp/dureader')
    # DuReaderData.statistics_paragraphs('../data/datasets/dureader')

    # DuReaderData.transform_segments('/sdc/zerozzl/temp/DuReader/preprocessed', '/sdc/zerozzl/temp/dureader')
    # DuReaderData.statistics_segments('../data/datasets/dureader')

    # DuReaderData.es_save_doc('../data/datasets/dureader_search', '10.79.169.35', 'dureader-search-doc', 'doc')
    # DuReaderData.es_save_doc('../data/datasets/dureader_zhidao', '10.79.169.35', 'dureader-zhidao-doc', 'doc')
    # DuReaderData.es_save_doc('../data/datasets/dureader_all', '10.79.169.35', 'dureader-all-doc', 'doc')

    # DuReaderData.es_save_paragraph('../data/datasets/dureader_search', '10.79.169.35', 'dureader-search-para', 'para')
    # DuReaderData.es_save_paragraph('../data/datasets/dureader_zhidao', '10.79.169.35', 'dureader-zhidao-para', 'para')
    # DuReaderData.es_save_paragraph('../data/datasets/dureader_all', '10.79.169.35', 'dureader-all-para', 'para')

    # DuReaderData.es_save_segment('../data/datasets/dureader/search', '10.79.169.35',
    #                              'dureader-search-seg', 'seg')
    # DuReaderData.es_save_segment('../data/datasets/dureader/zhidao', '10.79.169.35',
    #                              'dureader-zhidao-seg', 'seg')
    # DuReaderData.es_save_segment('../data/datasets/dureader/all', '10.79.169.35',
    #                              'dureader-all-seg', 'seg')

    # DuReaderData.create_segments('../data/datasets/dureader', '10.79.169.35', 'bigrams', debug=False)
    DuReaderData.create_dpr('../data/datasets/dureader', debug=False)
