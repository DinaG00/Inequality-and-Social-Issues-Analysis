import numpy as np
import pandas as pd
import EFA.efa as efa
import PCA.pca as pca
import factor_analyzer as fa
from GRAPHICS import graphics as g
import EFA.utils as utl


table = pd.read_csv('dataIN/inequalityOfIncome.csv', index_col=0, na_values=':')
table = table.drop(columns=['Unnamed: 6'])
print(table)

obsNume = table.index.values
varNume = table.columns.values
matrice_numerica = table.values

# replacing the missing values
X = utl.replaceNAN(matrice_numerica)
X_df = pd.DataFrame(data=X, index=obsNume, columns=varNume)
X_df.to_csv('./dataOUT/EFA.csv')

# computing Bartlett sphericity test
sphericityBartlett = fa.calculate_bartlett_sphericity(x=X_df)
print(sphericityBartlett)
if sphericityBartlett[0] > sphericityBartlett[1]:
    print('There is at least one common factor to be extracted!')
else:
    print('There are no common factors!')
    exit(-1)

# computing the Kaiser-Meyer-Olkin indices
kmo_idices= fa.calculate_kmo(x=X_df)
print(kmo_idices)

# KMO index
if kmo_idices[1] > 0.5:
    print('The initial variables can be expressed through factors!')
else:
    print('There is no factor to expressed the causal variables!')
    exit(-2)

vector = kmo_idices[0]
print(vector, type(vector))
matrix = vector[:, np.newaxis]
print(matrix, type(matrix))

# obtaining a pandas.DataFrame from the kmo matrix
kmo_df = pd.DataFrame(data=matrix, columns=['Kaiser-Meyer-Olkin Index'],
                    index=varNume)
print(kmo_df)

# calling the intensity of link function
g.linkIntensity(matrice=kmo_df,
                titlu='Kaiser-Meyer-Olkin Indices')
g.afisare()

# extracting factors
noOfSignificantFactors = 1
chi2TabMin = 1
for k in range(2, X.shape[1]):
    modelFA = fa.FactorAnalyzer(n_factors=k)
    modelFA.fit(X=X_df)
    commonFactors = modelFA.loadings_
    specificFactors = modelFA.get_uniquenesses()
    print(commonFactors)
    print(specificFactors)

    modelEFA = efa.EFA(matrice=X)
    chi2Calc, chi2Tab = modelEFA.calculTestBartlett(loadings=commonFactors,
                                epsilon=specificFactors)
    print(chi2Calc, chi2Tab)

    if np.isnan(chi2Calc) or np.isnan(chi2Tab):
        break
    if chi2Tab < chi2TabMin:
        chi2TabMin = chi2Tab
        noOfSignificantFactors = k

print('The number of significant factors we could extracted:',
      noOfSignificantFactors)

fitModelFA = fa.FactorAnalyzer(n_factors=noOfSignificantFactors)
fitModelFA.fit(X=X_df)
factorLoadings = fitModelFA.loadings_
factorLoadings_df = pd.DataFrame(data=factorLoadings,
                index=varNume,
                columns=('F'+str(k+1) for k in range(noOfSignificantFactors)))
print(factorLoadings_df)
g.corelograma(matrice=factorLoadings_df, titlu='Correlogram of factor loadings')
g.afisare()

# getting the eigenvalues from Factor Analyser
eigenvalues = fitModelFA.get_eigenvalues()
print(eigenvalues, type(eigenvalues))

# explained variance by the eigenvalues from the initial model
g.componentePrincipale(valoriProprii=eigenvalues[0],
        titlu='Explained variance by the eigenvalues from the initial model')

# explained variance by the eigenvalues from the factor models
g.componentePrincipale(valoriProprii=eigenvalues[1],
        titlu='Explained variance by the eigenvalues from from the factor models')
g.afisare()

# creating a PCA model with the initial data
modelPCA = pca.PCA(X)
Rxc = modelPCA.getRxc()
Rxc_df = pd.DataFrame(data=Rxc, index=varNume,
        columns=('C'+str(j+1) for j in range(Rxc.shape[1])))
print(Rxc_df)
g.corelograma(matrice=Rxc_df, dec=2,
            titlu='Correlogram of factor loadings from PCA')

# getting the scores (standardised principal components)
scores = modelPCA.getScoruri()
scores_df = pd.DataFrame(data=scores,
            index=obsNume,
            columns=('C'+str(j+1) for j in range(Rxc.shape[1])))
print(scores_df)
g.linkIntensity(matrice=scores_df, dec=2,
            titlu='Scores from Principal Components Analysis')
qualityObs = modelPCA.getCalObs()
qualityObs_df = pd.DataFrame(data=qualityObs,
            index=obsNume,
            columns=('C'+str(j+1) for j in range(Rxc.shape[1])))
g.linkIntensity(matrice=qualityObs_df, dec=2,
    titlu='Quality of observation representation on the axes of the principal components')

g.afisare()







