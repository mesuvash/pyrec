import numpy as np
import scipy.sparse
from sklearn.utils.extmath import randomized_svd


class PureSVD:

    def __init__(self, args):
        self.nfactor = args.nfactor
        self.U = None
        self.V = None

    def fit(self, train_input, train):
        U, sigma, VT = randomized_svd(train, self.nfactor)
        sigma = scipy.sparse.diags(sigma, 0)
        self.U = U * sigma
        self.V = VT.T

    def recommend_all(self, data):
        return np.dot(self.U, self.V.T)

    def recommend(self, users, data):
        ufactors = self.U[users, :]
        return np.dot(ufactors, self.V.T)

    @classmethod
    def validate(cls, data_loader,  params, eval_metric, mapk=20):

        from pyrec.recommender.modelArgs import WRMFArgs
        ks = params["ks"]
        # vtrain, vtest = validation_generator(train)
        scores = []
        vtrain_input, vtrain, vtest, _, _ = data_loader.generateValidation()
        for k in ks:
            args = WRMFArgs(None, k)
            model = PureSVD(args)
            model.fit(vtrain_input, vtrain)
            score = eval_metric(
                vtrain_input, vtrain, vtest, model, mapk)
            scores.append(score[0])
        scores = np.array(scores)
        kidx = np.unravel_index(
            np.ndarray.argmax(scores), scores.shape)[0]
        print kidx
        best_k = ks[kidx]
        best_score = scores[kidx]
        best_params = {}
        best_params["score"] = best_score
        best_params["arg"] = WRMFArgs(None, best_k)
        return best_params
