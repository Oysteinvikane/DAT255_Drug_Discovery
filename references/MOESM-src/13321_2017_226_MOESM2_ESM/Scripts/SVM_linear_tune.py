from sklearn.svm import SVC
import csv
import os
import numpy as np
from sklearn.metrics import roc_auc_score,recall_score,precision_score,f1_score
from  sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
import time


#=============================================================================================
"""
This script was used for optimizing hyperparameter-tuning for SVM with linear kernel
"""
#=============================================================================================

print "Now Running"
outFil=open("SVM_linear_results.csv","a")
outFil.write("nfolds,cost,kern,RAUC,Std_RAUC,MCC,Std_MCC,FSCORE,Std_FSCORE, Recall,Std_Recall,Precision, Std_Precision"+"\n")

Path="/Path_to folder/Train_Test files/"
print Path


def svm(X_Train,Y_Train,x_test,y_test,cost,kern):
    print cost,kern
    svm=SVC(C=cost, kernel=kern,probability=True)

    svm.fit(X_Train,Y_Train)

    y_pred= svm.predict(x_test)
    y_pred_score=svm.predict_proba(x_test)

    temp=[]
    for j in range(len(y_pred_score)):
    	temp.append(y_pred_score[j][1])
    acc2=accuracy_score(y_test, y_pred)
    mcc=matthews_corrcoef(y_test,y_pred)
    auc=roc_auc_score(np.array(y_test),np.array(temp))
    Recall=recall_score(y_test, y_pred,pos_label=1)
    Precision=precision_score(y_test, y_pred,pos_label=1)
    F1_score=f1_score(y_test, y_pred,pos_label=1)


    return acc2,auc,mcc,Recall,Precision,F1_score



def folds_SVM(path,nfolds,cost,kern):
    print
    print path
    print
    print cost,kern
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

        print
        print "Fold: ",(j+1)
        print "Now reading file: ",Trf
        Xreader=csv.reader(open(Trf,"rb"),delimiter=',')
        for row in Xreader:
            X_Train.append(np.array(row[1:],dtype=float))
            Y_Train.append(int(row[0]))
        print "Now reading file: ",Tesf
        xreader=csv.reader(open(Tesf,"rb"),delimiter=',')
        for row in xreader:
            x_test.append(np.array(row[1:],dtype=float))
            y_test.append(int(row[0]))

        acc2,auc,mcc,Recall,Precision,F1_score=svm(X_Train,Y_Train,x_test,y_test,cost,kern)

        print acc2,auc,mcc,Recall,Precision,F1_score

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


def SVM_parameters(path,nfolds,cost,kern):
    AvAUC,Std_AUC,AvMCC,Std_MCC,AvFSCORE, Std_FSCORE,AvRecall,Std_RECALL,AvPrecision, Std_PRECISION = folds_SVM(path,nfolds,cost,kern)
    outFil.write(str(nfolds)+","+str(cost)+","+str(kern)+","+str(AvAUC)+","+str(Std_AUC)+","+str(AvMCC)+","+str(Std_MCC)+","+str(AvFSCORE)+","+str(Std_FSCORE)+","+str(AvRecall)+","+str(Std_RECALL)+","+str(AvPrecision)+","+str(Std_PRECISION)+"\n")
    return

if __name__ == '__main__':

    #grid-search for hyperparameter-tuning
    SVM_parameters(Path,5,1000,'linear')
    SVM_parameters(Path,5,100,'linear')
    SVM_parameters(Path,5,10,'linear')
    SVM_parameters(Path,5,1,'linear')
    SVM_parameters(Path,5,0.1,'linear')
    SVM_parameters(Path,5,0.01,'linear')
    SVM_parameters(Path,5,0.001,'linear')
    SVM_parameters(Path,5,0.0001,'linear')
