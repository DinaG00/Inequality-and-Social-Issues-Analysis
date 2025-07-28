import numpy as np


class PCA:
    def __init__(self, X, regularizare=True):
        self.X = X
        self.Xstd = self.setModelStd()

        self.Cov = np.cov(m=self.Xstd, rowvar=False)
        valoriProprii, vectoriProprii = np.linalg.eigh(self.Cov)

        kReverse = [k for k in reversed(np.argsort(valoriProprii))]

        self.alpha = valoriProprii[kReverse]
        self.a = vectoriProprii[:, kReverse]

        if regularizare == True:
            for j in range(self.a.shape[1]):
                min = np.min(a=self.a[:, j])
                max = np.max(a=self.a[:, j])
                if np.abs(min) > np.abs(max):
                     self.a[:, j] = -self.a[:, j]

        self.C= self.Xstd @ self.a

        self.scores = self.C / np.sqrt(self.alpha)

        self.C2 = self.C * self.C

        self.Rxc = self.a * np.sqrt(self.alpha)


    def setModelStd(self):
        avgs = np.mean(a=self.X, axis=0)
        stds = np.std(a=self.X, axis=0)
        return (self.X - avgs) / stds

    def getModelStd(self):
        return self.Xstd

    def getEigenValues(self):
        return self.alpha

    def getEigenVectors(self):
        return self.a

    def getComponente(self):
        return self.C

    def getScores(self):
        return self.scores

    def getQualObs(self):
        sumForObs = np.sum(a=self.C2, axis=1)
        return np.transpose(self.C2.T / sumForObs)

    def getFactorLoadings(self):
        return self.Rxc

    def getObsContrib(self):
        return self.C2 / (self.alpha * self.X.shape[0])

    def getComunalitati(self):
        Rxc2 = self.Rxc * self.Rxc
        return np.cumsum(Rxc2, axis=1)




