import numpy as np
import os

def cal_kl(x, y, max_rcs, step_rcs):
    BINS = np.arange(max_rcs, 1.0, step_rcs)
    c1, _ = np.histogram(x, bins=BINS, density=False)
    c2, _ = np.histogram(y, bins=BINS, density=False)
    c1 = c1 + 0.00001
    c2 = c2 + 0.00001
    c1 = c1 / np.sum(c1)
    c2 = c2 / np.sum(c2)
    kl = np.sum(np.where(c1 != 0, c1 * np.log(c1 / c2), 0))
    return kl


def rcshr(predictions, dists, max_rcs, step_rcs, alpha, info_db, info_val, rcs_path):
    new_predictions = []
    new_dists = []
    for index in range(len(predictions)):
        prediction = predictions[index]
        dist = dists[index]

        scores = []
        query_index = index
        for predicition_index in prediction:
            score_sum = 0
            for bias in [0]:
                query_rcs_file = os.path.join(rcs_path, '{:0>5d}.bin'.format(bias + int(info_val[query_index][0])))
                query_rcs = np.fromfile(query_rcs_file, dtype=np.float64).reshape(-1, 1)
                pre_rcs_file = os.path.join(rcs_path, '{:0>5d}.bin'.format(bias + int(info_db[predicition_index][0])))
                pre_rcs = np.fromfile(pre_rcs_file, dtype=np.float64).reshape(-1, 1)
                score = cal_kl(query_rcs, pre_rcs, max_rcs=max_rcs, step_rcs=step_rcs)
                score_sum += score
            scores.append(score_sum / 1.0)
        scores = np.array(scores, dtype=np.float64)
        dist = np.array(dist, dtype=np.float64)
        prediction = np.array(prediction, dtype=np.float64)

        total_scores = alpha * scores + (1 - alpha) * dist
        ind = np.argsort(total_scores)
        prediction = prediction[ind]
        new_predictions.append(prediction)
        new_dists.append(total_scores[ind])
    return np.array(new_predictions), np.array(new_dists)

