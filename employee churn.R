

df=read.csv('Attrition.csv',header = T)
str(df)
summary(df)

#feature selection
install.packages('Boruta',dependencies = T)
library(mlbench)
library(randomForest)
library(Boruta)

set.seed(123)
boruta<-Boruta(Attrition~.,data = df, doTrace=2,maxRuns=200)
print(boruta)
dev.new()
plot(boruta,las=2,cex.axis=0.7)
dev.new()
plotImpHistory(boruta)

# data cleaning ( str )
df$EmployeeCount=NULL
df$Over18=NULL
df$DailyRate=NULL
df$MonthlyRate=NULL
df$Gender=NULL
df$EmployeeNumber=NULL
df$StandardHours=NULL
df$TrainingTimesLastYear=NULL
df$HourlyRate=NULL
df$PercentSalaryHike=NULL
df$BusinessTravel=NULL
df$Education=NULL
df$EducationField=NULL
df$PerformanceRating=NULL
df$RelationshipSatisfaction=NULL
df$WorkLifeBalance=as.factor(df$WorkLifeBalance)
df$StockOptionLevel=as.factor(df$StockOptionLevel)
df$EnvironmentSatisfaction=as.factor(df$EnvironmentSatisfaction)
df$JobInvolvement=as.factor(df$JobInvolvement)
df$JobSatisfaction=as.factor(df$JobSatisfaction)


str(df)

# data cleaning ( str )
library(plyr)

df$WorkLifeBalance <- revalue(df$WorkLifeBalance, 
                           c('1'='Bad','2'='Good','3'='Better','4'='Best'))

df$EnvironmentSatisfaction <- revalue(df$EnvironmentSatisfaction,
                                      c('1'='Low','2'='Medium','3'='High','4'='Very High'))
df$JobInvolvement <- revalue(df$JobInvolvement,
                             c('1'='Low','2'='Medium','3'='High','4'='Very High'))
df$JobSatisfaction <- revalue(df$JobSatisfaction,
                              c('1'='Low', '2'='Medium', '3'='High', '4'='Very High'))



str(df)
summary(df)


#check for NA's

sapply(df,function(x)sum(is.na(x)))

# EDA
install.packages('gmodels')
library(gmodels)


dev.new()
boxplot(df$MonthlyIncome~df$Attrition)


dev.new()
boxplot(df$YearsAtCompany~df$Attrition)

dev.new()
boxplot(df$YearsSinceLastPromotion~df$Attrition)

#Creating test and train data
y<-c()
n<-c()

for (x in 1:1470)
{
 if(df$Attrition[x]=="Yes")
  y<-c(x,y)
 else
  n<-c(x,n)
}


set.seed(123)
rnum<-sample(y,length(y)*0.7,replace = F)   
rnum1<-sample(y,length(y)*0.3,replace = F) 

#seperate 90% data for Training rest 10% for Testing
trny<-df[rnum,]  
tsty<-df[rnum1,]  

set.seed(123)
rnum<-sample(n,length(n)*0.8,replace = F)   
rnum1<-sample(n,length(n)*0.2,replace = F) 

#seperate 70% data for Training rest 30% for Testing
trnn<-df[rnum,]  
tstn<-df[rnum1,]  


tst<-rbind(tsty,tstn)
trn<-rbind(trny,trnn)




#Decision Tree
install.packages('rpart.plot',dependencies = T)
library(rpart.plot)
library(rpart)

dtree1=rpart(Attrition~.,data = trn,
             method = 'class')
# Tree plot
#install.packages("rattle",dependencies = T)
#library(rattle)

#dev.new()
#fancyRpartPlot(dtree1,type = 3)


# Predict & Confusion Matrix
tst$predProb=predict(dtree1,newdata = tst)
str(trn$Attrition)

tst$pred=ifelse(tst$predProb>0.5,'No','Yes')
str(tst$pred)
tst$pred=factor(tst$pred[1:317],levels = c('No','Yes'))

library(caret)
dtcm<-confusionMatrix(tst$pred,tst$Attrition)
dtcm



# Random Forest
rf <-randomForest(Attrition~.,data=trn) 
print(rf)

tst$rfPredProb= predict(rf, newdata=tst,type = 'prob')
rfPred= predict(rf, newdata=tst)

rfcm<-confusionMatrix(rfPred,tst$Attrition)
rfcm


#logistic regression
## use glm for logistical model 

attrlog=glm(Attrition~.,data = trn,family = "binomial")


tst$lpredProb=predict(attrlog,newdata=tst,type = "response")    #predicts chances of being 1


tst$lpred=ifelse(tst$lpredProb>0.5,'Yes','No')   #if probability is more than 0.5 convert into 1 or else if less than 0.5 convert into 0
str(tst$lpred)   #check str of new column 'pred'
tst$lpred=factor(tst$lpred,levels =c('No','Yes') )

lrcm<-confusionMatrix(tst$lpred,tst$Attrition) 
lrcm



####Comparing between the models
dtcm$table
lrcm$table
rfcm$table


#######Compare between the two models
# ROC curve & AUC
library(pROC)

dev.new(1)

plot.roc(tst$Attrition,tst$predProb[1:317],
         print.auc=T,  main="Decision Tree")

dev.new(2)

plot.roc(tst$Attrition,tst$rfPred[1:317],
         
         print.auc=T, main="Random Forest")

dev.new(3)

plot.roc(tst$Attrition,tst$lpredProb[1:317],
         
         print.auc=T, main="Logistic Regression")




#Since AUC of Random Forest > Decision Tree ,Logistic Regression
#also accurcy ofrandom forest is the best
#Random Forest is the Best Model




