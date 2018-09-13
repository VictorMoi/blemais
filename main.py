### Balbou et Victor

#V: Je fais un peu nawak, on reparlera organisation de code etc...

# chargement packages
import numpy as np
import pandas as pd

import os

#dossier="C:/Users/Victor/Documents/programmes/blemais/" 

project_path = os.getcwd()


# we load data

PDmaize = pd.read_table(project_path+"TrainingDataSet_Maize.txt")
colL = list(PDmaize)
col = {}
for val,key in enumerate(colL):
        col[key]=val


maize = np.array(maize)

colL.index("Tx_1")
col["Tx_1"]


# création de nouvelles variables

###  Déficit hydrique
RUM = 10

maize['RU_1']=RUM

for(i in 2:9){
  colRU<-paste0("RU_",i)
  colRU1<-paste0("RU_",i-1)
  colPR<-paste0("PR_",i)
  colETP<-paste0("ETP_",i)
  colETR<-paste0("ETR_",i)
  colDE<-paste0("DE_",i)
  DE<-(maize[,colPR]-maize[,colETP])<0
  maize[!DE,colRU]<-pmin(maize[!DE,colRU1]+maize[!DE,colPR]-maize[!DE,colETP],RUM)
  maize[DE,colRU]<-pmax(0,maize[DE,colRU1] * exp((maize[DE,colPR]-maize[DE,colETP])/RUM))
  maize[!DE,colETR]<-maize[!DE,colETP]
  maize[DE,colETR]<-maize[DE,colRU1]-maize[DE,colRU]+maize[DE,colPR]
  maize[,colDE]<-maize[,colETP]-maize[,colETR]
}








