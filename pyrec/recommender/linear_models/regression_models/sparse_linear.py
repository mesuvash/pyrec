from sklearn.linear_model import SGDRegressor
import scipy.sparse
import numpy as np
from pyrec.recommender.linear_models.regression_models.base import BaseLinear
from pyrec.parallel.ipythonParallelLinear import ParallelRunner


class SparseLinear(BaseLinear):

    """docstring for SparseLinear"""

    def __init__(self, arg):
        super(SparseLinear, self).__init__()
        self.arg = arg
        self.target = None
        self.__initargs()

    def __initargs(self):
        self.alpha = self.arg.l1 + self.arg.l2
        self.l1_ratio = self.arg.l1 / self.alpha
        self.loss = self.arg.loss
        self.penalty = self.arg.penalty
        self.intercept = self.arg.intercept
        self.model_type = self.arg.model_type

    def __getLearner(self):
        return SGDRegressor(loss=self.loss, penalty=self.penalty,
                            alpha=self.alpha, l1_ratio=self.l1_ratio,
                            fit_intercept=self.intercept)

    def fit(self, train_input, train_target, target_indices=None):
        import numpy as np
        import scipy.sparse
        models = []
        if self.model_type == "user":
            train_target = train_target.T
            train_input = train_input.T
        if target_indices is not None:
            train_target = train_target[:, target_indices]
        else:
            target_indices = range(train_target.shape[1])
        #for fast column access
        train_target = train_target.tocsc()
        for i, index in enumerate(target_indices):
            learner = self.__getLearner()
            y = np.ravel(train_target.getcol(i).todense())
            learner.fit(train_input, y)
            models.append(learner.coef_)
        self.sim = scipy.sparse.csc_matrix(np.vstack(models).T)
        return target_indices, self.sim

    def fit_parallel(self, train_input, train_target,
                     target_indices=None, num_procs=5,
                     batch_size=1000):
        prunner = ParallelRunner(self, num_procs, batch_size)
        indices, sim = prunner.fit(train_input, train_target)
        self.sim = sim
        return indices, self.sim

    def recommend_all(self, train_input):
        self.fit(data)
        if self.model_type == "item":
            score = train_input * self.sim
        elif self.model_type == "user":
            score = (self.sim.T * train_input).tocsr()
        return score

    def recommend(self, users, train_input):
        if self.model_type == "item":
            reco = train_input[users, :] * self.sim
        if self.model_type == "user":
            reco = (self.sim[:, users].T * train_input)
        return reco.todense()

    @classmethod
    def validate(cls, data_loader,  params, eval_metric, mapk=20):
        from pyrec.recommender.modelArgs import SparseLinearArgs

        l1s = params["l1"]
        l2s = params["l2"]
        mtypes = params["mtypes"]
        losses = params["loss"]
        penalties = params["penalty"]
        scores = []
        # vtrain, vtest = validation_generator(train)
        vtrain_input, vtrain, vtest, _, _ = data_loader.generateValidation()
        for loss in losses:
            penalty_scores = []
            for penalty in penalties:
                mtype_scores = []
                for mtype in mtypes:
                    l1_scores = []
                    for l1 in l1s:
                        l2_scores = []
                        for l2 in l2s:
                            args = SparseLinearArgs(
                                l1, l2, mtype, loss, penalty)
                            model = SparseLinear(args)
                            indices, sim = model.fit(vtrain_input, vtrain)
                            score = eval_metric(
                                vtrain_input, vtrain, vtest, model, mapk)
                            l2_scores.append(score[0])
                        l1_scores.append(l2_scores)
                    mtype_scores.append(l1_scores)
                penalty_scores.append(mtype_scores)
            scores.append(penalty_scores)
        scores = np.array(scores)
        lidx, pidx, midx, l1idx, l2idx = np.unravel_index(
            np.ndarray.argmax(scores), scores.shape)

        best_penalty = penalties[pidx]
        best_loss = losses[lidx]
        best_model,  best_l1, best_l2 = mtypes[
            midx], l1s[l1idx],  l2s[l2idx]
        best_score = scores[lidx, pidx, midx, l1idx, l2idx]
        best_params = {}
        best_params["score"] = best_score
        best_params["arg"] = SparseLinearArgs(best_l1, best_l2,
                                              best_model, best_loss,
                                              best_penalty)
        return best_params

    @classmethod
    def validate_parallel(cls, data_loader,  params, eval_metric,
                          mapk=20, num_procs=5, batch_size=1000):
        from pyrec.recommender.modelArgs import SparseLinearArgs
        l1s = params["l1"]
        l2s = params["l2"]
        mtypes = params["mtypes"]
        losses = params["loss"]
        penalties = params["penalty"]
        scores = []
        # vtrain, vtest = validation_generator(train)
        vtrain_input, vtrain, vtest, _, _ = data_loader.generateValidation()
        for loss in losses:
            penalty_scores = []
            for penalty in penalties:
                mtype_scores = []
                for mtype in mtypes:
                    l1_scores = []
                    for l1 in l1s:
                        l2_scores = []
                        for l2 in l2s:
                            args = SparseLinearArgs(
                                l1, l2, mtype, loss, penalty)
                            model = SparseLinear(args)
                            pmodel = ParallelRunner(
                                model, num_procs, batch_size)
                            indices, sim = pmodel.fit(vtrain_input, vtrain)
                            model.sim = sim
                            score = eval_metric(
                                vtrain_input, vtrain, vtest, model, mapk)
                            del model
                            del sim
                            l2_scores.append(score[0])
                        l1_scores.append(l2_scores)
                    mtype_scores.append(l1_scores)
                penalty_scores.append(mtype_scores)
            scores.append(penalty_scores)
        scores = np.array(scores)
        lidx, pidx, midx, l1idx, l2idx = np.unravel_index(
            np.ndarray.argmax(scores), scores.shape)

        best_penalty = penalties[pidx]
        best_loss = losses[lidx]
        best_model,  best_l1, best_l2 = mtypes[
            midx], l1s[l1idx],  l2s[l2idx]
        best_score = scores[lidx, pidx, midx, l1idx, l2idx]
        best_params = {}
        best_params["score"] = best_score
        best_params["arg"] = SparseLinearArgs(best_l1, best_l2,
                                              best_model, best_loss,
                                              best_penalty)
        return best_params
