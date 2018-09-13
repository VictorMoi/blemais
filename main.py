### Balbou et Victor

#V: Je fais un peu nawak, on reparlera organisation de code etc...

# chargement packages
import numpy as np
import pandas as pd

import os

#dossier="C:/Users/Victor/Documents/programmes/blemais/" 

project_path = os.getcwd()


# we load data

PDmaize = pd.read_table(os.path.join(project_path, "data", "TrainingDataSet_Maize.txt"))
maize = np.array(PDmaize)




colL = list(PDmaize)
col = {}
for val,key in enumerate(colL):
    col[key]=val



maize = np.array(maize)

colL.index("Tx_1")
col["Tx_1"]

PDmaize["Tx_1"]

maize[col["Tx_1"]]

# création de nouvelles variables

###  Déficit hydrique
RUM = 10



maize[col["RU_1"]]=RUM

for i in range(2,10):
    colRU = "RU_" + i
    colRU1 = "RU_" + (i - 1)
    colPR = "PR_" + i
    colETP = "ETP_" + i
    colETR = "ETR_" + i
    colDE = "DE_" + i

    DE = (maize[col[colPR]] - maize[col[colETP]]) < 0



  maize[!DE,colRU]<-pmin(maize[!DE,colRU1]+maize[!DE,colPR]-maize[!DE,colETP],RUM)
  maize[DE,colRU]<-pmax(0,maize[DE,colRU1] * exp((maize[DE,colPR]-maize[DE,colETP])/RUM))
  maize[!DE,colETR]<-maize[!DE,colETP]
  maize[DE,colETR]<-maize[DE,colRU1]-maize[DE,colRU]+maize[DE,colPR]
  maize[,colDE]<-maize[,colETP]-maize[,colETR]
}








