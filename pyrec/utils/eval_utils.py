from scipy import stats
import numpy as np
import math
import sys
from pyrec.evaluate.eval_ranking_metric import evalMetricsParallelMiniBatch, evalMetrics

from pyrec.utils.general_utils import force_print


def getConfidenceInterval(data, percent=0.95, distribution="t"):
    n, min_max, mean, var, skew, kurt = stats.describe(data)
    std = np.sqrt(var)
    if distribution == "t":
        R = stats.t.interval(percent, len(data) - 1, loc=mean,
                             scale=std / math.sqrt(len(data)))
    else:
        R = stats.norm.interval(
            percent, loc=mean, scale=std / math.sqrt(len(data)))
    error = (R[1] - R[0]) / 2
    return mean, error


def calculateCI(data, percent=0.95):
    d = np.array(data)
    if len(d.shape) == 1:
        return getConfidenceInterval(data, percent)
    else:
        k = d.shape[1]
        results = []
        for i in range(k):
            temp = d[:, i]
            score = getConfidenceInterval(temp, percent)
            results.append(score)
        return results


def getParallelEvalMetric(nprocs, batch_size):
    from pyrec.evaluate.eval_ranking_metric import evalMetricsParallelMiniBatch

    def _evalMetric(train_input, train_target, test, model, mapk):
        return evalMetricsParallelMiniBatch(train_input, train_target,
                                            test, model, mapk,
                                            batch_size=batch_size,
                                            nprocs=nprocs)
    return _evalMetric


def evaluate_fold(data_loader, nfolds, vparams, model_class,
                  eval_metrics, mapk, num_procs=1, batch_size=1000):

    iknn_precs = []
    iknn_recs = []
    iknn_maps = []

    for i in range(1, nfolds + 1):
        if i == 1:
            data_loader.setFoldIndex(i)
            force_print(
                "Running Validation: hyperparameter search on validation data")
            if num_procs <= 1:
                best_params = model_class.validate(data_loader, vparams,
                                                   eval_metrics,
                                                   mapk)
            else:
                best_params = model_class.validate_parallel(data_loader, vparams,
                                                            eval_metrics,
                                                            mapk, num_procs, batch_size)
            print best_params
            best_score = best_params["score"]
            best_arg = best_params["arg"]

            force_print("Best argument  : %s" % str(best_arg))
            force_print("Validation finished \nBest score : %g" % best_score)
        print best_arg
        data_loader.setFoldIndex(i)
        force_print("Evaluating fold : %d" % i)
        data_loader.transformer.update(best_params)
        train_input, train, test, ufeat, ifeat = data_loader.loadData()
        model = model_class(best_arg)
        if num_procs <= 1:
            model.fit(train_input, train)
        else:
            model.fit_parallel(train_input, train,
                               num_procs=num_procs,
                               batch_size=batch_size)
        _map, prec, rec, nusers = eval_metrics(train_input, train,
                                               test, model, mapk)
        del model
        del train_input
        del train
        del test
        iknn_maps.append(_map)
        iknn_precs.append(prec)
        iknn_recs.append(rec)
    print iknn_maps, iknn_precs, iknn_recs
    return calculateCI(iknn_maps), calculateCI(iknn_precs), calculateCI(iknn_recs), nusers
