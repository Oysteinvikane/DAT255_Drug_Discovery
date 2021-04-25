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



def knn(X_Train,Y_Train,x_test,y_test,Nn):

    knn= KNeighborsClassifier(n_neighbors=Nn)

    knn.fit(X_Train,Y_Train)
    y_pred= knn.predict(x_test)
    y_pred_prob=knn.predict_proba(x_test)

    temp=[]
    print y_pred_prob
    for j in range(len(y_pred_prob)):
	temp.append(y_pred_prob[j][1])

    auc=roc_auc_score(np.array(y_test),np.array(temp))
    acc2=accuracy_score(y_test,y_pred)
    mcc=matthews_corrcoef(y_test,y_pred)
    Recall=recall_score(y_test, y_pred,pos_label=1)
    Precision=precision_score(y_test, y_pred,pos_label=1)
    F1_score=f1_score(y_test, y_pred,pos_label=1)
    print auc,mcc,F1_score,Recall,Precision,acc2
    return y_test,y_pred_prob,y_pred,auc,acc2,mcc,Recall,Precision,F1_score



def kNN_predict(ifile1,ifile2,ofile1,ofile2,Nn):
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

    y_test,y_pred_prob,y_pred,auc,acc2,mcc,Recall,Precision,F1_score=knn(X_Train,Y_Train,x_test,y_test,Nn)

    writer=csv.writer(open(ofile1,"wb"),delimiter=',')
    writer2=csv.writer(open(ofile2,"wb"),delimiter=',')


    newHeaders=['auc','acc2','mcc','Recall','Precision','F1_score']
    print y_test
    temp=[]
    for j in range(len(y_pred_prob)):
    	temp.append(y_pred_prob[j][1])
    auc=roc_auc_score(np.array(y_test),np.array(temp))
    writer2.writerow(newHeaders)
    writer2.writerow([auc,acc2,mcc,Recall,Precision,F1_score])
    print "AUC",auc
    headers=['Y_true','Y_predict_NB']
    for j in range(len(y_test)):
        writer.writerow([y_test[j],temp[j]])



    return


if __name__ == '__main__':
    kNN_predict("/Path to trainfile set/train.csv",
    "/Path to validation set/validation.csv"
    ,"kNN_Validation_Predictions.csv","kNN_Validation_Metrics.csv",3)
