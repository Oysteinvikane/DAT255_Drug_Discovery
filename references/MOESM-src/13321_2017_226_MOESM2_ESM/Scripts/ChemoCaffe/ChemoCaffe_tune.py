"""
@Author: Keith Monaghan
@Date: 7/24/15
@Title: ChemoCaffe
@Description: Automates the testing of multiple network configurations in Caffe
"""

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
from  sklearn.metrics import matthews_corrcoef,auc,roc_auc_score,recall_score,precision_score,f1_score,accuracy_score
import csv
import numpy as np
import time
import os
import ConfigParser
from sys import argv

class CaffeNet:

	_test_size=0
	_batch_size=0
	_epochs=0
	_lr=0
	_test_interval=0
	_data_path= ""
	_solver=""
	_output=""
	_name=""
	_folds=0
	_train_size=0
	_test_size=0
	_numClasses=0
	#Constructor 
	#Initializes variables which are usually constant across different network configurations
	def __init__(self,name,folds,numClasses,epochs,test_interval,lr,batch_size,data,solver,output):
		self._folds=folds
		self._test_interval=test_interval
		self._data_path=data
		self._solver=solver
		self._output=output
		self._epochs=epochs
		self._lr=lr
		self._batch_size=batch_size
		self._numClasses=numClasses
		self._name=name
		self.getDataSize()
		writer=csv.writer(open(output,'a'),delimiter=',')
		writer.writerow(['Epochs','Layers','Activation','Input dropout','Dropout','L2','Filler','RAUC','RAUC_std','MCC','MCC_std','F1','F1_std','Recall','Recall_std','Precision','Precision_std','Acc','Acc_std'])
	#Helper Function for constructor
	#Reads size of test data set 
	#Converts epochs to iterations
	def getDataSize(self):
		reader1=csv.reader(open(self._data_path+'train_1.csv',"rU"))
		reader2=csv.reader(open(self._data_path+'test_1.csv',"rU"))
		self._train_size = 0
		self._test_size = 0
		for row in reader1:
			self._train_size = self._train_size+1
		for row in reader2:
			self._test_size = self._test_size+1

		self._epochs = (self._epochs*self._train_size)/self._batch_size
		self._test_interval = (self._test_interval*self._train_size)/self._batch_size

	#Builds the network with specified hyperparameters
	def MakeNetwork(self,db,batch_size,layers,deploy,act,input_dropout,hidden_dropout,L2,filler):
		
		#Create Data layer
		data, label = L.HDF5Data(source=db,batch_size=batch_size,ntop=2)
	
		#Add hidden layers
		top = data
		if(input_dropout!=0):
			top = L.Dropout(top, in_place=True, dropout_ratio = input_dropout)
		
		test = 0
		for x in range(0,len(layers)):
			if(L2):
				if(filler==1):
					top = L.InnerProduct(top, num_output=layers[x], weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'),param=[dict(decay_mult=1)])
				elif(filler==2):
					top = L.InnerProduct(top, num_output=layers[x], weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type='gaussian',std=0.01),param=[dict(decay_mult=1)])

			else:
				if(filler==1):
					top = L.InnerProduct(top, num_output=layers[x], weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'),param=[dict(decay_mult=0)])
				elif(filler==2):
					top = L.InnerProduct(top, num_output=layers[x], weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type='gaussian',std=0.01),param=[dict(decay_mult=0)])

	
			if(act == 1):
				top = L.ReLU(top,in_place=True)
			elif(act == 2):
				top = L.Sigmoid(top, in_place=True)
			elif(act == 3):
				top = L.TanH(top, in_place=True)
			else:
				print "Error, invalid activation function choice "
			if(hidden_dropout!=0):
				top = L.Dropout(top, in_place=True, dropout_ratio = hidden_dropout)
	
		#Add Output Layers
		if(filler==1):
			output = L.InnerProduct(top, num_output=self._numClasses,weight_filler=dict(type='xavier'),bias_filler=dict(type='xavier'))
		elif(filler==2):
			output = L.InnerProduct(top, num_output=self._numClasses,weight_filler=dict(type='gaussian',std=0.01),bias_filler=dict(type='gaussian',std=0.01))

		if(deploy == False):
			loss = L.SoftmaxWithLoss(output,label)
			return to_proto(loss)	
		else:
			prob = L.Softmax(output)
			return to_proto(prob)
	
	#Generates train, test, and deploy network definitions
	#Writes networks to .prototxt format to be read by Caffe
	def WritePrototxt(self,layers,act,input_dropout,hidden_dropout,L2,filler):
		with open('train.prototxt','w') as f:
			print >>f, self.MakeNetwork('train.txt',self._batch_size,layers,False,act,input_dropout,hidden_dropout,L2,filler)
		with open('test.prototxt','w') as f:
			print >>f, self.MakeNetwork('test.txt',self._batch_size,layers,False,act,0,0,L2,filler)
		with open('deploy.prototxt','w') as f:
			print >>f, self.MakeNetwork('test.txt',self._test_size,layers,True,act,0,0,L2,filler)	
	
	#Trains and evaluates the performance of a network across multiple folds
	def testConfig(self,layers,act,input_dropout,hidden_dropout,L2,filler):

		#Generate .prototxt files
		self.WritePrototxt(layers,act,input_dropout,hidden_dropout,L2,filler)
		#Enable GPU Mode
		caffe.set_mode_gpu()

		MODEL_FILE = 'deploy.prototxt'

		#Set iters & learning rate in solver file
		with open('solver_new.prototxt','w') as new_file:
			with open(self._solver,'r') as old_file:
				for line in old_file:
					if 'max_iter:' in line:
						new_file.write('max_iter:'+str(self._epochs)+"\n")
					elif 'snapshot:' in line:
						new_file.write('snapshot:'+str(self._test_interval)+"\n")
					elif 'base_lr:' in line:
						new_file.write('base_lr:'+str(self._lr)+"\n")
					elif 'snapshot_prefix:' in line:
						new_file.write('snapshot_prefix:"'+self._name+'"\n')
					else:
						new_file.write(line)
		os.rename('solver_new.prototxt',self._solver)

		temp = []

		for i in range(0,self._folds):

			#Write text files specifying the databse path
			print "Writing txt files"
			with open('train.txt', 'w') as f:
				f.write(self._data_path+'train_' + str(i+1) + '.h5' + '\n')
			with open('test.txt', 'w') as f:
				f.write(self._data_path+'test_' + str(i+1) + '.h5' + '\n')
			print "Done"

			#Get Solver
			solver = caffe.SGDSolver(self._solver)
	
			#Train network
			solver.solve()

			for x in range(0,int(self._epochs/self._test_interval)):
				PRETRAINED = self._name+'_iter_' + str((x+1)*self._test_interval) + '.caffemodel'

				#Load network
				net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

				#Forward data
				net.forward()

				#Record results
				results = net.blobs['Softmax1'].data

				#Read test labels
				test_size = 0
				y_test=[]
				reader=csv.reader(open(self._data_path + 'test_'+str(i+1)+'.csv',"rU"),delimiter=',')
				for row in reader:
					y_test.append(int(row[0]))
					test_size = test_size+1

				#Read predictions
				y_pred = []
				y_pred_scores = []
				for y in range(0,test_size):
					y_pred.append(int(results[y].argmax()))
					y_pred_scores.append(float(results[y][1]))

				#Score Predictions
				acc=accuracy_score(y_test, y_pred)
				mcc=matthews_corrcoef(y_test,y_pred)
				RAUC=roc_auc_score(y_test,y_pred_scores)
				Recall=recall_score(y_test, y_pred,pos_label=1)
				Precision=precision_score(y_test, y_pred,pos_label=1)
				F1_score=f1_score(y_test, y_pred,pos_label=1)

				#Store scores in temporary list
				temp.append([self._test_interval*(x+1),acc,mcc,RAUC,Recall,Precision,F1_score])


		#Open output writer	in append mode	
		writer=csv.writer(open(self._output,'a'),delimiter=',')

		#Average results for each test interval and print to output file
		for y in range(0,self._epochs/self._test_interval):
			epochs=0
			acc=[]
			mcc=[]
			RAUC=[]
			Recall=[]
			Precision=[]
			F1_score=[]
			for x in range(y,len(temp),self._epochs/self._test_interval):
				epochs=((temp[x][0]*self._batch_size)/self._train_size)
				acc.append(float(temp[x][1]))
				mcc.append(float(temp[x][2]))
				RAUC.append(float(temp[x][3]))
				Recall.append(float(temp[x][4]))
				Precision.append(float(temp[x][5]))
				F1_score.append(float(temp[x][6]))
			
			#Write output
			acts = ["ReLU","Sigmoid","TanH"]
			fillers = ["Xavier","Gaussian"]
			
			AvAUC=sum(RAUC)/float(self._folds)
			Std_AUC=np.std(RAUC)
			AvMCC=sum(mcc)/float(self._folds)
			Std_MCC=np.std(mcc)
			AvFSCORE=sum(F1_score)/float(self._folds)
			Std_FSCORE=np.std(F1_score)
			AvRecall=sum(Recall)/float(self._folds)
			Std_Recall=np.std(Recall)
			AvPrecision=sum(Precision)/float(self._folds)
			Std_Precision=np.std(Precision)
			AvAccuracy=sum(acc)/float(self._folds)	
			Std_Accuracy=np.std(acc)

			writer.writerow([epochs+1,str(layers),acts[act-1],str(input_dropout),str(hidden_dropout),str(L2),str(fillers[filler-1]),str(AvAUC),str(Std_AUC),str(AvMCC),str(Std_MCC),str(AvFSCORE),str(Std_FSCORE),str(AvRecall),str(Std_Recall),str(AvPrecision),str(Std_Precision),str(AvAccuracy),str(Std_Accuracy)])

if __name__ == '__main__':
	t1=time.time()

	config = ConfigParser.RawConfigParser()
	config.read(argv[1])
	
	try:
		test=CaffeNet(config.get('DEFAULT','name'),config.getint('DEFAULT','folds'),config.getint('DEFAULT','number of classes'),config.getint('DEFAULT','epochs'),config.getint('DEFAULT','test interval'), config.getfloat('DEFAULT','learning rate'),config.getint('DEFAULT','batch size'), config.get('DEFAULT','data path'), config.get('DEFAULT','solver path'), config.get('DEFAULT','output path'))
	except ValueError:
		raise Exception('Value Error in DEFAULT section')
	
	if not os.path.exists(config.get('DEFAULT','temp path')):
		os.makedirs(config.get('DEFAULT','temp path'))
	
	os.chdir(config.get('DEFAULT','temp path'))

	#Check config file for errors
	for section in config.sections():
		try:
			assert int(config.get(section,'activation function')) >= 1 and int(config.get(section,'activation function'))<=3, ('Invalid choice for '+section+'\'s activation function.  Value must be 1 for ReLU, 2 for Sigmoid, or 3 for TanH.')
			assert int(config.get(section,'filler')) == 1 or int(config.get(section,'filler'))==2, ('Invalid choice for '+section+'\'s filler.  Value must be 1 for Xavier or 2 for Gaussian')
			assert float(config.get(section,'input dropout'))>=0.0 and float(config.get(section,'input dropout')) <= 1.0, ('Invalid ratio for '+section+'\'s input dropout ratio.  Value must be between 0 and 1')
			assert float(config.get(section,'hidden dropout'))>=0.0 and float(config.get(section,'hidden dropout')) <= 1.0, ('Invalid ratio for '+section+'\'s hidden dropout ratio.  Value must be between 0 and 1 (Recommended: 0.5)')
			assert config.get(section,'l2 regularization').lower() == 'false' or config.get(section,'l2 regularization').lower() == 'true',  ('Invalid choice for '+section+'\'s l2 regularization field.  Must be true or false')
			assert bool(config.get(section,'name')), ('Error: No name for ' + section)
			assert float(config.get(section,'learning rate')) > 0, 'Invalid choice for '+section+'\'s learning rate.  Value must be positive. (Recommended: 0.01)'
			assert int(config.get(section,'batch size')) > 0, 'Invalid choice for '+section+'\'s batch size.  Value must be positive. (Recommended: 256)'
			assert int(config.get(section,'epochs')) > 0, 'Invalid choice for '+section+'\'s # of epochs.  Value must be positive. (Recommended: 100)'
			assert int(config.get(section,'test interval')) > 0, 'Invalid choice for '+section+'\'s # of epochs.  Value must be positive. (Recommended: 10)'
			assert float(config.get(section,'weight decay')) > 0 and float(config.get(section,'weight decay')) < 1, 'Invalid choice for '+section+'\'s weight decay.  Value must be between 0 and 1 (Recommended: 0.0005)'
			assert bool(config.get(section,'name')), "Error: No name given"
			layers = [int(n) for n in config.get(section,'network').split(',')]
			for n in layers:
				assert n>0, 'Invalid # of neurons for '+section+'.  # of neurons must be positive.'

			for n in range(0,config.getint(section,'folds')):
				assert os.path.isfile(config.get(section,'data path')+'train_'+str((n+1))+'.csv'), 'Error fetching data.  Only '+n+' folds found'
				assert os.path.isfile(config.get(section,'data path')+'test_'+str((n+1))+'.csv'), 'Error fetching data.  Only '+n+' folds found'
			
			assert os.path.isfile(config.get(section,'solver path'))

			break

		except AssertionError, e:
			raise Exception(e.args[0])
		except ValueError, e:
			raise Exception(e.args[0] + ' in section '+section)

	for section in config.sections():
		reconstruct = False
		for option in config.options(section):
			if(config.has_option('DEFAULT',option) and (config.get('DEFAULT', option) != config.get(section, option))):
				reconstruct = True

		if(reconstruct):
			test=CaffeNet(config.getint('DEFAULT','folds'),config.getint('DEFAULT','number of classes'),config.getint('DEFAULT','epochs'),config.getint('DEFAULT','test interval'), config.getfloat('DEFAULT','learning rate'),config.getint('DEFAULT','batch size'), config.get('DEFAULT','data path'), config.get('DEFAULT','solver path'), config.get('DEFAULT','output path'))

		test.testConfig([int(n) for n in config.get(section,'network').split(',')],int(config.get(section,'activation function')),float(config.get(section,'input dropout')),float(config.get(section,'hidden dropout')),config.get(section,'l2 regularization').lower()=='true',int(config.get(section,'filler')))

	t2=time.time()
	print "Time Elapsed: ",t2-t1