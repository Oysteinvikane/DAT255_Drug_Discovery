from sklearn.neighbors import KNeighborsClassifier
import csv
import os
import numpy as np
from sklearn.metrics import auc,roc_auc_score,recall_score,precision_score,f1_score
from  sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time
import string

outFil=open("OutputFile.csv","a")
outFil.write("nfolds,Nn,RAUC,Std_RAUC,MCC,Std_MCC,FSCORE,Std_FSCORE, Recall,Std_Recall,Precision, Std_Precision"+"\n")

Path="/Path to train-test splits/"
"""
File format of files should be: Label,Feat1,Feat2,..., FeatN
"""

def knn(X_Train,Y_Train,x_test,y_test,Nn):

    knn= KNeighborsClassifier(n_neighbors=Nn)

    knn.fit(X_Train,Y_Train)
    y_pred= knn.predict(x_test)
    y_pred_prob=knn.predict_proba(x_test)

    temp=[]
    for j in range(len(y_pred_prob)):
        temp.append(y_pred_prob[j][1])
    auc=roc_auc_score(np.array(y_test),np.array(temp))
    acc2=accuracy_score(y_test,y_pred)
    mcc=matthews_corrcoef(y_test,y_pred)
    Recall=recall_score(y_test, y_pred,pos_label=1)
    Precision=precision_score(y_test, y_pred,pos_label=1)
    F1_score=f1_score(y_test, y_pred,pos_label=1)

    return auc,acc2,mcc,Recall,Precision,F1_score


def folds_KNN(path,nfolds,Nn):
    print
    print path
    print
    print "Number of Nearest neighbors considered: ",Nn
    RF_para=[]
    os.chdir(path)
    AUC=[]
    Acc=[]
    MCC=[]
    FSCORE=[]
    RECALL=[]
    PRECISION=[]
    for j in range(nfolds):
        Trf="train_"+str(j+1)+".csv"
        Tesf="test_"+str(j+1)+".csv"

        X_Train=[]
        Y_Train=[]
        x_test=[]
        y_test=[]
        countTrain=0
        countTest=0
        print
        print "Fold: ",(j+1)
        print "Now reading file: ",Trf
        Xreader=csv.reader(open(Trf,"rb"),delimiter=',')
        for row in Xreader:
            X_Train.append(np.array(row[1:],dtype=float))
            Y_Train.append(int(row[0]))
            countTrain+=1
        print "Now reading file: ",Tesf
        xreader=csv.reader(open(Tesf,"rb"),delimiter=',')
        for row in xreader:
            countTest+=1
            x_test.append(np.array(row[1:],dtype=float))
            y_test.append(int(row[0]))

        print "Training Samples, ",countTrain
        print "Test Samples", countTest
        auc,acc2,mcc,Recall,Precision,F1_score=knn(X_Train,Y_Train,x_test,y_test,Nn)


        print auc,acc2,mcc,Recall,Precision,F1_score

        AUC.append(float(auc))
        Acc.append(float(acc2))
        MCC.append(float(mcc))
        FSCORE.append(float(F1_score))
        RECALL.append(float(Recall))
        PRECISION.append(float(Precision))

    AvAUC=sum(AUC)/float(nfolds)
    Std_AUC=np.std(AUC)
    AvMCC=sum(MCC)/float(nfolds)
    Std_MCC=np.std(MCC)
    AvFSCORE=sum(FSCORE)/float(nfolds)
    Std_FSCORE=np.std(FSCORE)
    AvRecall=sum(RECALL)/float(nfolds)
    Std_Recall=np.std(RECALL)
    AvPrecision=sum(PRECISION)/float(nfolds)
    Std_Precision=np.std(PRECISION)
    print "ROC_AUC: ",AvAUC
    print "MCC: ",AvMCC
    print "F_Score: ",AvFSCORE
    print "Recall: ",AvRecall
    print "Precision: ",AvPrecision
    return AvAUC,Std_AUC,AvMCC,Std_MCC,AvFSCORE, Std_FSCORE,AvRecall,Std_Recall,AvPrecision, Std_Precision


def Knn_parameters(path,nfolds,Nn):

    AvAUC,Std_AUC,AvMCC,Std_MCC,AvFSCORE, Std_FSCORE,AvRecall,Std_RECALL,AvPrecision, Std_PRECISION=folds_KNN(path,nfolds,Nn)
    outFil.write(str(nfolds)+","+str(Nn)+","+str(AvAUC)+","+str(Std_AUC)+","+str(AvMCC)+","+str(Std_MCC)+","+str(AvFSCORE)+","+str(Std_FSCORE)+","+str(AvRecall)+","+str(Std_RECALL)+","+str(AvPrecision)+","+str(Std_PRECISION)+"\n")
    return

if __name__ == '__main__':
    #5-Fold Cross validation was applied
    Knn_parameters(Path,5,1)
    Knn_parameters(Path,5,3)
    Knn_parameters(Path,5,5)
    Knn_parameters(Path,5,7)
    Knn_parameters(Path,5,9)
    Knn_parameters(Path,5,11)