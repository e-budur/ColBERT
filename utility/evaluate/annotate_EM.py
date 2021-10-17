from multiprocessing import get_context
import os
import sys
import git
import tqdm
import ujson
import random

from argparse import ArgumentParser
from multiprocessing import Pool

from colbert.utils.utils import print_message, load_ranking, groupby_first_item
from utility.utils.qa_loaders import load_qas_, load_collection_
from utility.utils.save_metadata import format_metadata, get_metadata
from utility.evaluate.annotate_EM_helpers import *

# TODO: Tokenize passages in advance, especially if the ranked list is long! This requires changes to the has_answer input, slightly.

def main(args):
    qas = load_qas_(args.qas)
    collection = load_collection_(args.collection, retain_titles=True)
    print("collection[0]", collection[0])
    print("collection[1]", collection[1])
    try:
        rankings = load_ranking(args.ranking, types=[int, int, int, float, int])
    except:
        try:
            rankings = load_ranking(args.ranking, types=[int, int, int, int])
        except:
            try:
               rankings = load_ranking(args.ranking, types=[int, int, int])
            except:
               rankings = load_ranking(args.ranking, types=[int, int, int, str, int, int])
    
    with get_context("spawn").Pool() as parallel_pool:

        print_message('#> Tokenize the answers in the Q&As in parallel...')
        #qas = list(parallel_pool.map(tokenize_all_answers, qas))
        qas = list(tqdm.tqdm(parallel_pool.imap(tokenize_all_answers, qas), total=len(qas)))
        qid2answers = {qid: {'question':"", 'tok_answers': tok_answers} for qid, question, tok_answers in qas}
        assert len(qas) == len(qid2answers), (len(qas), len(qid2answers))

        print_message('#> Lookup passages from PIDs...')
        expanded_rankings = [(qid, pid, rank, collection[pid]['passage'], collection[pid]['docid'], qid2answers[qid]['question'], qid2answers[qid]['tok_answers'])
                            for qid, pid, rank, *_ in rankings]

        print_message('#> Assign labels in parallel...')
        #labeled_rankings = list(parallel_pool.map(assign_label_to_passage, enumerate(expanded_rankings)))
        labeled_rankings = list(tqdm.tqdm(parallel_pool.imap(assign_label_to_passage, expanded_rankings), total=len(expanded_rankings)))
        

    # Dump output.
    print_message("#> Dumping output to", args.output, "...")
    qid2rankings = groupby_first_item(labeled_rankings)

    num_judged_queries, num_ranked_queries = check_sizes(qid2answers, qid2rankings)

    # Evaluation metrics and depths.
    success, counts = compute_and_write_labels(args.output, qid2answers, qid2rankings, args.topk, args.output_question_passage_pairs, args.lang)

    # Dump metrics.
    with open(args.output_metrics, 'w') as f:
        d = {'num_ranked_queries': num_ranked_queries, 'num_judged_queries': num_judged_queries}

        extra = '__WARNING' if num_judged_queries != num_ranked_queries else ''
        d[f'success{extra}'] = {k: v / num_judged_queries for k, v in success.items()}
        d[f'counts{extra}'] = {k: v / num_judged_queries for k, v in counts.items()}
        d['arguments'] = get_metadata(args)

        f.write(format_metadata(d) + '\n')

    print('\n\n')
    print(args.output)
    print(args.output_metrics)
    print("#> Done\n")


if __name__ == "__main__":
    random.seed(12345)

    parser = ArgumentParser(description='.')

    # Input / Output Arguments
    parser.add_argument('--qas', dest='qas', required=True, type=str)
    parser.add_argument('--collection', dest='collection', required=True, type=str)
    parser.add_argument('--ranking', dest='ranking', required=True, type=str)
    parser.add_argument('--output_question_passage_pairs', dest='output_question_passage_pairs', default=False, action='store_true')
    parser.add_argument('--lang', dest='lang', required=False, type=str, default='tr')
    parser.add_argument('--topk', dest='topk', required=False, type=int, default=1000)

    args = parser.parse_args()

    args.output = f'{args.ranking}.annotated'
    args.output_metrics = f'{args.ranking}.annotated.metrics'

    assert not os.path.exists(args.output), args.output

    main(args)
