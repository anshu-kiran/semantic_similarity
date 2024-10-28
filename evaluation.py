from ast import literal_eval

from nltk.translate.bleu_score import sentence_bleu

from rouge import Rouge
import pandas as pd

eval_metric = pd.read_excel('results/eval_metrics.xlsx')


def str_from_list(strs):
    return ' '.join(strs)


def get_rouge(n, reference, candidate):
    rouge = Rouge()
    return rouge.get_scores(candidate, reference, avg=True)


def get_bleu(n, reference, candidate):
    return sentence_bleu([reference], candidate)


def get_exact_match(n, reference, candidate):
    return int(reference == candidate)


tot_rouge_r = 0
tot_rouge_p = 0
tot_rouge_f = 0
tot_bleu = 0
for (idx, row) in eval_metric.iterrows():
    reference = row['Gold label'].replace('[', '').replace(']', '')
    # print(reference)
    lda = literal_eval(row['LDA'])
    nmf = literal_eval(row['NMF'])
    lsa = literal_eval(row['LSA'])

    metric_used = lda

    # n = 1
    candidate1 = metric_used[0]
    candidate2 = str_from_list(metric_used[:2])
    candidate5 = str_from_list(metric_used[:5])
    candidate10 = str_from_list(metric_used[:10])
    candidate15 = str_from_list(metric_used)

    # tot_bleu += get_bleu(1, reference.split(' '), [candidate1])
    tot_bleu += get_bleu(15, reference.split(' '), metric_used[:15])
    tot_rouge_r += get_rouge(15, reference, candidate15)['rouge-l']['r']
    tot_rouge_p += get_rouge(15, reference, candidate15)['rouge-l']['p']
    tot_rouge_f += get_rouge(15, reference, candidate15)['rouge-l']['f']
print(f'AVG BLEU: {tot_bleu / 50}')
print(f'AVG ROUGE-R: {tot_rouge_r / 50}')
print(f'AVG ROUGE-P: {tot_rouge_p / 50}')
print(f'AVG ROUGE-F: {tot_rouge_f / 50}')
