### Balbou et Victor

#V: Je fais un peu nawak, on reparlera organisation de code etc...

# chargement packages
import numpy as np
import pandas as pd
import os

if os.name == 'posix':
    project_path = os.getcwd()
else:
    project_path = 

    
# we load data
def loadData(filename):
    PDmaize = pd.read_table(os.path.join(project_path, "data", filename))
    ind2name = list(PDmaize)
    name2ind = {i:j for j,i in enumerate(ind2name)}
    return np.array(PDmaize), ind2name, name2ind

maize, ind2name, name2ind = loadData("TrainingDataSet_Maize.txt")


def addColumn(arr, ind2name, name2ind, name, column):
    ind2name.append(name)
    name2ind[name] = len(ind2name)
    arr = np.concatenate(arr, column, axis=1)
    return arr, ind2name, name2ind




maize[name2ind["Tx_1"]]

# création de nouvelles variables

###  Déficit hydrique
RUM = 10



maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, "RU_1", RUM)

maize[col["RU_1"]]=RUM

for i in range(2,10):
    colRU = "RU_" + i
    colRU1 = "RU_" + (i - 1)
    colPR = "PR_" + i
    colETP = "ETP_" + i
    colETR = "ETR_" + i
    colDE = "DE_" + i

    DE = (maize[col[colPR]] - maize[col[colETP]]) < 0


#
#  maize[!DE,colRU]<-pmin(maize[!DE,colRU1]+maize[!DE,colPR]-maize[!DE,colETP],RUM)
#  maize[DE,colRU]<-pmax(0,maize[DE,colRU1] * exp((maize[DE,colPR]-maize[DE,colETP])/RUM))
#  maize[!DE,colETR]<-maize[!DE,colETP]
#  maize[DE,colETR]<-maize[DE,colRU1]-maize[DE,colRU]+maize[DE,colPR]
#  maize[,colDE]<-maize[,colETP]-maize[,colETR]
#}
#







