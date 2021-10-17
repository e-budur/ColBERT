from colbert.utils.utils import print_message
from utility.utils.dpr import DPR_normalize, has_answer


def tokenize_all_answers(args):
    qid, question, answers = args
    return qid, question, [DPR_normalize(ans) for ans in answers]


def assign_label_to_passage(args):
    (qid, pid, rank, passage, docid, question, tokenized_answers) = args

    # if idx % (1*1000*1000) == 0:
    #     print(idx)

    #return (qid, pid, rank, has_answer(tokenized_answers, passage), question, passage[:min(20, len(passage))], docid, '|'.join([' '.join(tokenized_answer) for tokenized_answer in tokenized_answers]))
    return (qid, pid, rank, has_answer(tokenized_answers, passage), question, "", "", "")


def check_sizes(qid2answers, qid2rankings):
    num_judged_queries = len(qid2answers)
    num_ranked_queries = len(qid2rankings)

    print_message('num_judged_queries =', num_judged_queries)
    print_message('num_ranked_queries =', num_ranked_queries)

    if num_judged_queries != num_ranked_queries:
        assert num_ranked_queries <= num_judged_queries

        print('\n\n')
        print_message('[WARNING] num_judged_queries != num_ranked_queries')
        print('\n\n')
    
    return num_judged_queries, num_ranked_queries


def compute_and_write_labels(output_path, qid2answers, qid2rankings, topk, output_question_passage_pairs, lang='tr'):
    cutoffs = [1, 5, 10, 20, 30, 50, 100, 1000, 'all']
    success = {cutoff: 0.0 for cutoff in cutoffs}
    counts = {cutoff: 0.0 for cutoff in cutoffs}

    with open(output_path, 'w') as f:
        for qid in qid2answers:
            if qid not in qid2rankings:
                continue

            prev_rank = 0  # ranks should start at one (i.e., and not zero)
            labels = []

            #for pid, rank, label, question, passage, docid, answer_texts in qid2rankings[qid][:topk]:
            for pid, rank, label, question, passage, docid, answer_texts in qid2rankings[qid]:
                assert rank == prev_rank+1, (qid, pid, (prev_rank, rank))
                prev_rank = rank

                labels.append(label)
                line_data = [qid, pid, rank, int(label)]
                if output_question_passage_pairs:
                    line_data.append(question.replace('\t', ' ').replace('\n', ' '))
                    line_data.append(answer_texts.replace('\t', ' ').replace('\n', ' '))
                    line_data.append(passage.replace('\t', ' ').replace('\n', ' '))
                    
                    line_data.append('http://{}.wikipedia.org/wiki?curid={}'.format(lang, docid))
                    
                line = '\t'.join(map(str, line_data)) + '\n'
                f.write(line)

            for cutoff in cutoffs:
                if cutoff != 'all':
                    success[cutoff] += sum(labels[:cutoff]) > 0
                    counts[cutoff] += sum(labels[:cutoff])
                else:
                    success[cutoff] += sum(labels) > 0
                    counts[cutoff] += sum(labels)

    return success, counts


# def dump_metrics(f, nqueries, cutoffs, success, counts):
#     for cutoff in cutoffs:
#         success_log = "#> P@{} = {}".format(cutoff, success[cutoff] / nqueries)
#         counts_log = "#> D@{} = {}".format(cutoff, counts[cutoff] / nqueries)
#         print('\n'.join([success_log, counts_log]) + '\n')

#         f.write('\n'.join([success_log, counts_log]) + '\n\n')
