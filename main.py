### Balbou et Victor

#V: Je fais un peu nawak, on reparlera organisation de code etc...

# chargement packages
import numpy as np
import pandas as pd
import os

if os.name == 'posix':
    project_path = os.getcwd()
else:
    project_path = 'C:/Users/Victor/Documents/programmes/Github/blemais'

    
# we load data
def loadData(filename):
    PDmaize = pd.read_table(os.path.join(project_path, "data", filename))
    ind2name = list(PDmaize)
    name2ind = {i:j for j,i in enumerate(ind2name)}
    return np.array(PDmaize), ind2name, name2ind

maize, ind2name, name2ind = loadData("TrainingDataSet_Maize.txt")


def addColumn(arr, ind2name, name2ind, name, column):
    column=np.transpose(np.array([column]))
    ind2name.append(name)
    name2ind[name] = len(ind2name)-1
    arr = np.concatenate((arr, column), axis=1)
    return arr, ind2name, name2ind




maize[name2ind["Tx_1"]]

# création de nouvelles variables

###  Déficit hydrique
RUM = 10





maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, "RU_1", np.array([RUM for i in range(len(maize))]))

for i in range(2,10):
    colRU = "RU_" + str(i)
    colRU1 = "RU_" + str(i - 1)
    colPR = "PR_" + str(i)
    colETP = "ETP_" + str(i)
    colETR = "ETR_" + str(i)
    colDE = "DE_" + str(i)

    DE = (maize[:,name2ind[colPR]] - maize[:,name2ind[colETP]]) < 0

    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, colRU, np.minimum(maize[:,name2ind[colRU1]] + maize[:,name2ind[colPR]] - maize[:,name2ind[colETP]],RUM)*(1 - DE) + DE*np.maximum(maize[:,name2ind[colRU1]]*np.exp((maize[:,name2ind[colPR]] - maize[:,name2ind[colETP]])/RUM),0))
    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, colETR, DE*(maize[:,name2ind[colRU1]] + maize[:,name2ind[colPR]] - maize[:,name2ind[colRU]]) + (1-DE)*(maize[:,name2ind[colETP]]))
    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, colDE, maize[:,name2ind[colETP]] - maize[:,name2ind[colETR]])

