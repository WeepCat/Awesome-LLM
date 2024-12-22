import evaluate
from transformers import EvalPrediction
acc = evaluate.load("accuracy", cache_dir="/root/autodl-tmp/cache/")


def accuracy(eval_pred: EvalPrediction, compute_result=False):
    # 计算评估指标，整合每个batch的结果
    # result = {}
    # pos_predictions_scores = eval_pred.predictions[0]
    # neg_predictions_scores = eval_pred.predictions[1]
    # # We assume that the first sample is preferred by default in groundtruth
    # result['accuracy'] = np.sum(
    #     pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    # return result
    if not compute_result:
        # 每个批次的预测结果添加到 metric 中
        pos_scores, neg_scores = eval_pred.predictions
        labels = (pos_scores > neg_scores)
        acc.add_batch(predictions=labels, references=[1] * len(labels))
        return {}

    else:
        # 汇总并返回最终结果
        result = acc.compute()
        return {"accuracy": result["accuracy"]}