from sklearn.svm import SVC
import csv
import os
import numpy as np
from sklearn.metrics import auc,roc_auc_score,recall_score,precision_score,f1_score
from  sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time


def svm(X_Train,Y_Train,x_test,y_test,cost,Gam,kern):
    print cost,Gam,kern
    svm=SVC(C=cost, kernel=kern, degree=3, gamma=float(Gam), coef0=0.0, shrinking=True, probability=True, tol=0.001,
            cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)

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


    return y_pred,temp,auc,mcc,F1_score,Recall,Precision,acc2


def svm_predict(ifile1,ifile2,ofile1,ofile2,cost,Gam,kern):
    print "Now reading file: ",ifile1
    Xreader=csv.reader(open(ifile1,"rb"),delimiter=',')
    X_Train=[]
    Y_Train=[]
    countTrain=0
    for row in Xreader:
        X_Train.append(np.array(row[1:],dtype=float))
        Y_Train.append(int(row[0]))
        countTrain+=1
    print "Number of training examples ",countTrain
    print
    print "Now reading file: ",ifile2
    xreader=csv.reader(open(ifile2,"rb"),delimiter=',')

    x_test=[]
    y_test=[]
    countTest=0
    for row in xreader:
        countTest+=1
        x_test.append(np.array(row[1:],dtype=float))
        y_test.append(int(row[0]))

    print "Training Samples, ",countTrain
    print "Test Samples", countTest
    print len(x_test)
    print len(y_test)

    y_pred,temp,auc,mcc,F1_score,Recall,Precision,acc2=svm(X_Train,Y_Train,x_test,y_test,cost,Gam,kern)

    writer=csv.writer(open(ofile1,"wb"),delimiter=',')
    writer2=csv.writer(open(ofile2,"wb"),delimiter=',')


    newHeaders=['auc','mcc','F1_score','Recall','Precision','acc2']
    print y_test
    print temp
    auc=roc_auc_score(np.array(y_test),np.array(temp))
    writer2.writerow(newHeaders)
    writer2.writerow([auc,mcc,F1_score,Recall,Precision,acc2])
    print "AUC",auc
    headers=['Y_true','Y_predict_NB']
    for j in range(len(y_test)):
        writer.writerow([y_test[j],temp[j]])

    return


if __name__ == '__main__':
    svm_predict("/Path to training file /train_test.csv",
    "/Path to validation file/validation.csv"
    ,"SVMrbf_Validation_Predictions.csv","SVM_rbf_Validation_Metrics.csv",100,0.01,"rbf")
