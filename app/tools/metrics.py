import re

def rouge_n(n_gram=2, ref_summarys=None, candidate_summary=None):
    """
    rouge_n 摘要测评
    :param n_gram:
    :param ref_summary: 参考摘要集 []
    :param candidate_summary: 候选摘要 str
    :return: float
    """
    # 我在北京  我在 再北 北京'
    pattern = r'[\w+]'
    candidate_summary = re.findall(pattern, candidate_summary)
    candidate_summary = [w for w in candidate_summary]
    candidate_n_gram = [''.join(candidate_summary[i: i + n_gram]) if i + n_gram < len(candidate_summary)
                        else candidate_summary[i] for i, s in enumerate(candidate_summary)]
    ref_summarys = [re.findall(pattern, summary) for summary in ref_summarys]
    ref_summarys_n_gram = [[''.join(summary[i: i+ n_gram]) if i + n_gram < len(summary) else summary[i]
                            for i, s in enumerate(summary)] for summary in ref_summarys]
    repeats_n_gram = [set(candidate_n_gram).intersection(set(summary_n_gram))
                     for summary_n_gram in ref_summarys_n_gram]
    precision_score = [len(v) / len(set(candidate_n_gram)) for i, v in enumerate(repeats_n_gram)]
    recall_score = [len(v) / len(set(ref_summarys_n_gram[i])) for i, v in enumerate(repeats_n_gram)]
    for i, v in enumerate(repeats_n_gram):
        n = len(v)
        s = len(set(ref_summarys_n_gram[i]))
    return precision_score, recall_score
