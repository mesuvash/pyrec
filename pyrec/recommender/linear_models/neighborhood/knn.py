import numpy as np
import scipy.sparse
import progressbar
from pyrec.utils.distance_metric.similarity import keepTopKSimilarity


class KnnReco(object):

    def __init__(self, modelargs):
        self.similarity = modelargs.similarity
        self.k = modelargs.k
        self.model_type = modelargs.model_type

    # train_target is dummy; added to be consistednt with API
    def fit(self, train_input, train_target=None):

        if self.model_type == "item":
            self.sim = self.similarity(train_input)
        elif self.model_type == "user":
            self.sim = self.similarity(train_input.T.tocsr())
        if self.k is not None:
            top_k = keepTopKSimilarity(self.sim, self.k)
            del self.sim
            self.sim = top_k

    def recommend_all(self, data):
        self.fit(data)
        if self.model_type == "item":
            score = data * self.sim
        elif self.model_type == "user":
            score = (self.sim.T * data).tocsr()
        return score

    def recommend(self, users, data):
        if self.model_type == "item":
            reco = data[users, :] * self.sim
        if self.model_type == "user":
            reco = (self.sim[:, users].T * data)
        return reco.todense()

    @classmethod
    def validate(cls, data_loader,  params, eval_metric, mapk=20):
        # def validate(cls, train, params, eval_metric, validation_generator,
        # transformer=None, mapk=20):
        from pyrec.recommender.modelArgs import KnnArgs
        ks = params["ks"]
        ks.sort(reverse=True)
        sim_fns = params["sims"]
        mtypes = params["mtypes"]

        scores = []
        vtrain_input, vtrain, vtest, _, _ = data_loader.generateValidation()
        for mtype in mtypes:
            sim_scores = []
            for sim_fn in sim_fns:
                args = KnnArgs(sim_fn, None, mtype)
                model = KnnReco(args)
                model.fit(vtrain_input)
                neigh_scores = []
                for k in ks:
                    top_k = keepTopKSimilarity(model.sim, k)
                    model.sim = top_k
                    score = eval_metric(
                        vtrain_input, vtrain, vtest, model, mapk)
                    neigh_scores.append(score[0])
                sim_scores.append(neigh_scores)
            scores.append(sim_scores)
        scores = np.array(scores)
        midx, simidx, kidx = np.unravel_index(
            np.ndarray.argmax(scores), scores.shape)
        best_model, best_sim, best_k = mtypes[midx], sim_fns[simidx], ks[kidx]
        best_score = scores[midx, simidx, kidx]
        best_params = {}
        best_params["k"] = best_k
        best_params["score"] = best_score
        best_params["arg"] = KnnArgs(best_sim, best_k, best_model)
        return best_params