from sklearn.naive_bayes import BernoulliNB
import csv
import os
import numpy as np
from sklearn.metrics import auc,roc_auc_score,recall_score,precision_score,f1_score
from  sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time
import string

outFil=open("/Path to results/NB_Optimization.csv","a")
outFil.write("nfolds,alpha,binarize,class_prior,fit_prior,AvAUC,Std_AUC,AvMCC,Std_MCC,AvFSCORE,Std_FSCORE,AvRecall,Std_Recall,AvPrecision,Std_Precision"+"\n")

Path="/Path to train-test splits/TrainTest/"



def BernoulNB(X_train, Y_train, x_test, y_test,a,binz,cl_pr,ft_pr):
	nb = BernoulliNB(alpha=float(a),binarize=float(binz),class_prior=cl_pr,fit_prior=ft_pr)
	nb.fit(X_train, Y_train)
	y_pred= nb.predict(x_test)
	y_pred_prob=nb.predict_proba(x_test)
	temp=[]
	for j in range(len(y_pred_prob)):
		temp.append(y_pred_prob[j][1])
	auc=roc_auc_score(np.array(y_test),np.array(temp))
	#acc2=accuracy_score(y_test,y_pred)
	mcc=matthews_corrcoef(y_test,y_pred)
	Recall=recall_score(y_test, y_pred,pos_label=1)
	Precision=precision_score(y_test, y_pred,pos_label=1)
	F1_score=f1_score(y_test, y_pred,pos_label=1)
		
	return auc,mcc,Recall,Precision,F1_score



def folds_NB(path,nfolds,a,binz,cl_pr,ft_pr):
	print
	print path
	print
	print a,binz,cl_pr,ft_pr
	os.chdir(path)
	AUC=[]
	MCC=[]
	FSCORE=[]
	RECALL=[]
	PRECISION=[]
	for j in range(nfolds):
		print j
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
		auc,mcc,Recall,Precision,F1_score=BernoulNB(X_Train,Y_Train,x_test,y_test,a,binz,cl_pr,ft_pr)
		print auc,mcc,Recall,Precision,F1_score
		AUC.append(float(auc))
		MCC.append(float(mcc))
		FSCORE.append(float(F1_score))
		RECALL.append(float(Recall))
		PRECISION.append(float(Precision))
		print MCC

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

	return AvAUC,Std_AUC,AvMCC,Std_MCC,AvFSCORE,Std_FSCORE,AvRecall,Std_Recall,AvPrecision,Std_Precision

def NB_parameters(path,nfolds,a,binz,cl_pr,ft_pr):
	AvAUC,Std_AUC,AvMCC,Std_MCC,AvFSCORE,Std_FSCORE,AvRecall,Std_Recall,AvPrecision,Std_Precision=folds_NB(path,nfolds,a,binz,cl_pr,ft_pr)
	metr=[]
	metr=AvAUC,Std_AUC,AvMCC,Std_MCC,AvFSCORE,Std_FSCORE,AvRecall,Std_Recall,AvPrecision,Std_Precision
	stringA=[]
	for j in range(len(metr)):
		stringA.append(str(metr[j]))
	outFil.write(str(nfolds)+","+str(a)+","+str(binz)+","+str(cl_pr)+","+str(ft_pr)+","+string.join(stringA,",")+"\n")
	return


if __name__ == '__main__':
	#Tested configurations
	#Path to data
	NB_parameters(Path,5,1.0,0.0,None,True)
	NB_parameters(Path,5,0.5,0.0,None,True)
	NB_parameters(Path,5,0.1,0.0,None,True)
	NB_parameters(Path,5,1.0,0.0,None,False)
	NB_parameters(Path,5,0.5,0.0,None,False)
	NB_parameters(Path,5,0.1,0.0,None,False)	

