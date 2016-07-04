import numpy as np
import scipy.sparse


class MF:

    def __init__(self, args):
        self.l2 = args.l2
        self.nfactor = args.nfactor
        self.U = None
        self.V = None

    def initParams(self, m, n, U=None, V=None):
        if U is None and self.U is None:
            self.U = np.random.randn(m, self.nfactor) * 0.01
        else:
            self.U = U
        if V is None and self.V is None:
            self.V = np.random.randn(n, self.nfactor) * 0.01
        else:
            self.V = V

    def update(self, X, R):
        I = scipy.sparse.identity(self.nfactor, format="csr")
        preinv = np.dot(X.T, X) + self.l2 * I
        inv = np.linalg.inv(preinv)
        X_R = X.T * R
        return np.dot(inv, X_R).T

    def updateU(self, train):
        self.U = self.update(self.V, train.T)

    def updateV(self, train):
        self.V = self.update(self.U, train)

    def fit(self, train_input, train, n_iter=10, U=None, V=None):
        m, n = train.shape
        self.initParams(m, n, U, V)
        for _ in xrange(n_iter):
            self.updateU(train)
            self.updateV(train)

    def recommend_all(self, data):
        return np.dot(self.U, self.V.T)

    def recommend(self, users, data):
        ufactors = self.U[users, :]
        return np.dot(ufactors, self.V.T)

    @classmethod
    def validate(cls, data_loader,  params, eval_metric, mapk=20):
        # def validate(cls, train, params,  eval_metric, validation_generator,
        # transformer, mapk=20):
        from pyrec.recommender.modelArgs import WRMFArgs
        ks = params["ks"]
        lamdas = params["lamdas"]
        # vtrain, vtest = validation_generator(train)
        scores = []
        vtrain_input, vtrain, vtest, _, _ = data_loader.generateValidation()
        for k in ks:
            lamda_scores = []
            for c in lamdas:
                args = WRMFArgs(c, k)
                model = MF(args)
                model.fit(vtrain_input, vtrain)
                score = eval_metric(
                    vtrain_input, vtrain, vtest, model, mapk)
                lamda_scores.append(score[0])
            scores.append(lamda_scores)
        print scores
        scores = np.array(scores)
        kidx, cidx = np.unravel_index(
            np.ndarray.argmax(scores), scores.shape)
        best_k, best_lamda = ks[kidx],  lamdas[cidx]
        best_score = scores[kidx, cidx]
        best_params = {}
        best_params["score"] = best_score
        best_params["arg"] = WRMFArgs(best_lamda, best_k)
        return best_params
