import numpy as np
import PCA.pca as pca
import scipy.stats as sts


class EFA:

    def __init__(self, matrice):
        self.X = matrice

        pcaModel = pca.PCA(self.X)
        self.Xstd = pcaModel.getXstd()
        self.Corr = pcaModel.getCorr()
        self.ValProp = pcaModel.getValProp()
        self.Scoruri = pcaModel.getScoruri()
        self.CalObs = pcaModel.CalObs

    def getXstd(self):
        return self.Xstd

    def getValProp(self):
        return self.ValProp

    def getScoruri(self):
        return self.Scoruri

    def getCalObs(self):
        return self.CalObs

    def calculTestBartlett(self, loadings, epsilon):
        n = self.X.shape[0]
        m, q = np.shape(loadings)
        print(n, m, q)
        V = self.Corr
        psi = np.diag(v=epsilon)
        VE = loadings @ loadings.T + psi
        VE_1 = np.linalg.inv(a=VE)
        IE = VE_1 @ V
        detIE = np.linalg.det(a=IE)
        if detIE > 0:
            chi2Calc = (n-1 - (2*m + 4*q - 5) / 6) * \
                            (np.trace(a=IE) - np.log(detIE) - m)
            df = ((m - q)**2 - m - q) / 2
            chi2Tab = 1 - sts.chi2.cdf(chi2Calc, df)
        else:
            chi2Calc, chi2Tab = np.NaN, np.NaN

        return chi2Calc, chi2Tab