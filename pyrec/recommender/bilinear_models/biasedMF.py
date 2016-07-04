import numpy as np
import scipy.sparse


class MF:

    def __init__(self, args):
        self.l2 = args.l2
        self.nfactor = args.nfactor
        self.U = None
        self.V = None
        self.Uf = None
        self.If = None

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

    def fit(self, train, n_iter=10, U=None, V=None, Uf=None, If=None):
        m, n = train.shape
        self.initParams(m, n, U, V, )
        for i in xrange(n_iter):
            self.updateV(train)
            self.updateU(train)

    def recommend_all(self, data):
        return np.dot(self.U, self.V.T)

    def recommend(self, users, data):
        ufactors = self.U[users, :]
        return np.dot(ufactors, self.V.T)

class FeatureMF:

    def __init__(self, args):
        self.l2 = args.l2
        self.nfactor = args.nfactor
        self.U = None
        self.V = None
        self.Wu = None
        self.Wv = None

    def initParams(self, Uf, If, U=None, V=None):

        m = Uf.shape[0]
        n = If.shape[0]
        if U is None and self.U is None:
            self.U = np.random.randn(m, self.nfactor) * 0.01
        else:
            self.U = U
        if V is None and self.V is None:
            self.V = np.random.randn(n, self.nfactor) * 0.01
        else:
            self.V = V
        self.Wu = np.random.randn(m, If.shape[1]) * 0.01
        self.Wi = np.random.randn(n, Uf.shape[1]) * 0.01

    def update(self, X, Xf, R, Yf, Wy):
#         print X.shape, Xf.shape, R.shape, Yf.shape, Wy.shape
        Z = np.hstack([X, Xf])
        I = scipy.sparse.identity(Z.shape[1], format="csr")
        preinv = np.dot(Z.T, Z) + self.l2 * I
        inv = np.linalg.inv(preinv)
        inv_Z = np.dot(inv, Z.T)
        D = np.dot(np.dot(Z.T, Wy), Yf.T)

        result = inv_Z * R.T - np.dot(inv, D) 
        return result.T
    
    def updateU(self, train, Uf, If):
        result = self.update(self.V, self.If, train, Uf, self.Wi)
        self.U, self.Wu = result[:, :self.U.shape[1]], result[:, self.U.shape[1]:]

    def updateV(self, train, Uf, If):
        result = self.update(self.U, self.Uf, train.T, If, self.Wu)
        self.V, self.Wi = result[:, :self.V.shape[1]], result[:, self.V.shape[1]:]

        
    def fit(self, train, Uf, If, n_iter=10, U=None, V=None):
        self.initParams( Uf, If, U, V)
        self.Uf = Uf
        self.If = If
        for _ in xrange(n_iter):
            self.updateU(train, Uf, If)
            self.updateV(train, Uf, If)
 
    def recommend_all(self, data):
        return np.dot(self.U, self.V.T) + np.dot(self.Uf,self.Wi.T) + np.dot(self.Wu, self.If.T )

    def recommend(self, users, data, Uf, If):
        return np.dot(self.U[users, :], self.V.T) + \
                np.dot(self.Uf[users, :], self.Wi.T) +\
                np.dot(self.Wu[users,:], self.If.T)


# import numpy as np
# import scipy.sparse


# class BiasedMF:

#     def __init__(self, args):
#         self.l2 = args.l2
#         self.nfactor = args.nfactor
#         self.U = None
#         self.V = None

#     def initParams(self, m, n, U=None, V=None):
#         if U is None and self.U is None:
#             self.U = np.random.randn(m, self.nfactor + 1) * 0.01
#         else:
#             self.U = U
#         if V is None and self.V is None:
#             self.V = np.random.randn(n, self.nfactor + 1) * 0.01
#         else:
#             self.V = V

#     def update(self, X, R, other_bias):
#         I = scipy.sparse.identity(self.nfactor + 1, format="csr")
#         preinv = np.dot(X.T, X) + self.l2 * I
#         inv = np.linalg.inv(preinv)
#         invX = np.dot(inv, X.T)

#         result = invX * R - np.dot()
#         return np.dot(inv, X_R).T

#     def updateU(self, train):
#         temp = self.V[:, 0].copy()
#         self.V[:, 0] = 1.0
#         self.U = self.update(self.V, train.T, temp)
#         self.V[:, 0] = temp

#     def updateV(self, train):
#         temp = self.U[:, 0].copy()
#         self.U[:, 0] = 1.0
#         self.V = self.update(self.U, train, temp)
#         self.U[:, 0] = temp

#     def fit(self, train, n_iter=10, U=None, V=None):
#         m, n = train.shape
#         self.initParams(m, n, U, V)

#         for _ in xrange(n_iter):
#             self.updateU(train)
#             self.updateV(train)
#         self.fixbias()

#     def fixbias(self):
#         self.U = np.insert(self.U, 0, 1, axis=1)
#         self.V = np.insert(self.V, 1, 1, axis=1)

#     def recommend_all(self, data):
#         return np.dot(self.U, self.V.T)

#     def recommend(self, users, data):
#         ufactors = self.U[users, :]
#         return np.dot(ufactors, self.V.T)

