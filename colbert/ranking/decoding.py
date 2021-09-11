import os
import codecs
from typing import OrderedDict
import jsonlines
from colbert.utils.parser import Arguments
import csv
import json
from tqdm import tqdm
from collections import defaultdict
import random

def read_answers(args):
    print("reading answers")
    with codecs.open(args.squad_file, mode='r', encoding='utf-8') as squad_file:
        squad_data = json.load(squad_file)

    answers_dict = OrderedDict()

    for squad_data_item in tqdm(squad_data['data']):
        for paragraph in squad_data_item['paragraphs']:            
            for qa in paragraph['qas']:
                qa_id = qa['id']
                answers_tr = qa['answers_tr']
                answers_dict[qa_id] = answers_tr

    return answers_dict

def read_queries(args):
    print("reading queries")
    with codecs.open(args.queries, encoding="utf-8") as queries_file:
        queries_dict = OrderedDict()
        for row_index, line in enumerate(queries_file):
            fields = line.strip().split("\t")
            qid = fields[0]
            q_text = fields[1]
            queries_dict[qid] = {
                'qid': qid,
                'text': q_text
            }
        return queries_dict

def read_collection(args):
    print("reading collections")
    with codecs.open(args.collection, encoding="utf-8") as collection_file:
        collection_dict = OrderedDict()
        for p_row_index, line in enumerate(collection_file):
            fields = line.strip().split("\t")
            pid = fields[0]
            contents = ""
            if len(fields) == 2:
                contents = fields[1]
            collection_dict[p_row_index] = {
                "pid": pid,
                "contents": contents
            }
        return collection_dict

def read_rankings(args):
    print("reading rankings")
    with codecs.open(args.rankings, encoding="utf-8") as ranking_file:
        ranking_dict = OrderedDict()
        for line in ranking_file:
            qid, p_row_index, *_ = line.strip().split("\t")
            if qid not in ranking_dict:
                ranking_dict[qid] = []
            
            p_doc = args.collection[int(p_row_index)]
            ranking_dict[qid].append(p_doc)
        return ranking_dict

def prepare_triples(args):
    print("preparing triples")
    positive_results = defaultdict(list)
    negative_results = defaultdict(list)
    for qid in tqdm(args.queries.keys()):
        q_rankings = args.rankings[qid]

        for doc in q_rankings:
            pid = doc['pid']
            contents = doc['contents']
            answer_texts = list(set([answer['text'] for answer in args.answers[qid]]))

            is_positive = False
            if len(positive_results[qid]) <= 3:
                for answer_text in answer_texts:
                    if len(positive_results[qid]) <= 3 and answer_text in contents:
                        positive_results[qid].append(pid)
                        is_positive = True
                        break
            
            if is_positive == False:
                is_found = False
                for answer_text in answer_texts:
                    if answer_text in contents:
                        is_found = True    
                if is_found == False:
                    negative_results[qid].append(pid)
    
    with jsonlines.open(args.output_triples, mode="w") as triples_file:
        for qid in tqdm(args.queries.keys()):
            if len(positive_results[qid]) == 0 or len(negative_results[qid]) == 0:
                continue 
            
            selected_negative_results = [random.choice(negative_results[qid]) for i in positive_results[qid]]
            for positive_result, negative_result in zip(positive_results[qid], selected_negative_results):
                triples_file.write([qid, positive_result, negative_result])
                        


def decode(args):
    args.queries = read_queries(args)
    args.collection = read_collection(args)
    args.rankings = read_rankings(args)
    args.answers = read_answers(args)
    prepare_triples(args)
    print('\n\n')
    print(args.output_triples)
    print("#> Done.")
    print('\n\n')

if __name__ == "__main__":
    parser = Arguments(description='Ranking file decoder to prepare triples.')
    parser.add_argument('--queries', dest='queries', required=True)
    parser.add_argument('--rankings', dest='rankings', required=True)
    parser.add_argument('--collection', dest='collection', required=True)
    parser.add_argument('--squad_file', dest='squad_file', required=True)
    parser.add_argument('--output_triples', dest='output_triples', required=True)
    args = parser.parse()

    decode(args)