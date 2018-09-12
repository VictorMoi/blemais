library(ggplot2)
library(nlme)
library(lmer)

setwd("C:/Users/Victor/Documents/programmes/blemais/")


maize<-read.csv("./TrainingDataSet_Maize.txt",sep="\t")

wheat<-read.csv("./TestDataSet_Wheat_blind.txt",sep="\t")
hist(wheat$PR_1)

hist(maize$yield_anomaly)

colnames(maize)

ggplot()+
  geom_boxplot(data=maize,aes(x=as.factor(year_harvest),y=yield_anomaly))

ggplot()+
  geom_boxplot(data=maize,aes(x=as.factor(NUMD),y=yield_anomaly))

ggplot()+
  geom_boxplot(data=maize,aes(x=as.factor(IRR),y=yield_anomaly))

RUM<-10

maize$RU_1<-RUM

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

colnames(maize)

mean(maize$DE_2)
mean(maize$DE_3)
mean(maize$DE_4)
mean(maize$DE_5)
mean(maize$DE_6)
mean(maize$DE_7)
mean(maize$DE_8)
mean(maize$DE_9)

mean(maize$RU_8)

for(i in 1:9){
  moy<-paste0("Tm_",i)
  min<-paste0("Tn_",i)
  max<-paste0("Tx_",i)
  maize[,moy]<-(maize[,min]+maize[,max])/2
}

  
maize$GDD49<-pmax(maize$Tm_4-5,0)*30+
  pmax(maize$Tm_5-5,0)*31+
  pmax(maize$Tm_6-5,0)*30+
  pmax(maize$Tm_7-5,0)*31+
  pmax(maize$Tm_8-5,0)*31+
  pmax(maize$Tm_9-5,0)*30
  
lm.maize1<-lm(yield_anomaly~NUMD+year_harvest+IRR+RV_5+RV_6+RV_7+RV_8+RV_9+GDD49+DE_5+DE_6+DE_7+DE_8+DE_9+Tn_5+Tn_6+SeqPR_5+SeqPR_6+SeqPR_7+SeqPR_8+SeqPR_9,data=maize)
summary(lm.maize1)

lm.maize2<-lm(yield_anomaly~year_harvest+IRR+RV_5+RV_7+RV_8+RV_9+GDD49+DE_5+DE_7+DE_8+Tn_5+Tn_6+SeqPR_5+SeqPR_6+SeqPR_7+SeqPR_9,data=maize)
summary(lm.maize2)
error<-(maize$yield_anomaly-predict(lm.maize2))
sqrt(mean(error^2))
error<-(maize$yield_anomaly-predict(lm.maize1))
sqrt(mean(error^2))
