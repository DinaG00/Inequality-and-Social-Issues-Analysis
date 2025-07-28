import pandas as pd
import PCA.pca as pca
from GRAPHICS import graphics as graph

# Load the dataset
table = pd.read_csv('./dataIN/inequalityOfIncome.csv', index_col=0)
table = table.drop(columns=['Unnamed: 6'])
print(table)

# Getting the columns for PCA (excluding the first column)
cols = table.columns.values[1:].tolist()
print(cols, type(cols))

# Getting the observations (row indices)
obs = table.index.values.tolist()
print(obs, type(obs))

# Extracting the numpy ndarray for PCA
X = table[cols].values
print(X, X.shape)

# PCA model
pcaModel = pca.PCA(X)

# Standardized model
stdTable = pcaModel.getModelStd()

# Saving the standardized model matrix to a CSV file
stdTable_df = pd.DataFrame(data=stdTable, columns=cols, index=obs)
stdTable_df.to_csv(path_or_buf='./dataOUT/PCA.csv', index_label='Region Code')

# Creating a graph showing the variance explained by the principal components
graph.principalComponents(eigenvalues=pcaModel.getEigenValues())
graph.show()

# Extracting the quality of representation of the observations
qualObs = pcaModel.getQualObs()
qualObs_df = pd.DataFrame(data=qualObs, columns=('C'+str(j+1) for j in range(len(cols))),
                          index=obs)

# Creating a graph for the quality of representation of the observations on the principal component axes
graph.linkIntensity(qualObs_df, title='Quality of Observation Representation on Principal Component Axes')
graph.show()

# Labels for the principal components
component_labels = ['C'+str(j+1) for j in range(len(cols))]

# Graphical representation of the scores
scores = pcaModel.getScores()
scores_df = pd.DataFrame(data=scores, columns=component_labels,
                         index=obs)
graph.linkIntensity(matrix=scores_df, title='Scores')
graph.show()

# Graphical representation of the contributions of observations to the variance of the principal component axes
obsContrib = pcaModel.getObsContrib()
obsContrib_df = pd.DataFrame(data=obsContrib, columns=component_labels,
                             index=obs)
graph.linkIntensity(matrix=obsContrib_df,
                   title='Contribution of Observations to the Variance of Principal Component Axes')
graph.show()

# Extracting the factor loadings
Rxc = pcaModel.getFactorLoadings()
Rxc_df = pd.DataFrame(data=Rxc, columns=component_labels,
                      index=cols)
graph.correlogram(matrix=Rxc_df)
graph.show()

# Representation in the correlation circle of causal variables in the field of the first two principal components
graph.correlCircle(matrix=Rxc_df)
graph.show()

# Extracting communalities and create a correlogram of the communalities
comun = pcaModel.getComunalitati()
comun_df = pd.DataFrame(data=comun, columns=component_labels,
                        index=cols)
graph.correlogram(matrix=comun_df, title='Correlogram of Communalities')
graph.show()
