import re
from rouge_score import rouge_scorer

def compute_rouge_score(hypothesis: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return (
        scores["rouge1"].fmeasure
        + scores["rouge2"].fmeasure
        + scores["rougeL"].fmeasure
    ) / 3.0

hyp = "[658, 460, 706, 596]"
ref = "[658, 460, 706, 596]"
print("Same:", compute_rouge_score(hyp, ref))

ref2 = "[616, 539, 683, 604]"
print("Diff:", compute_rouge_score(hyp, ref2))
