import sys
from sys import argv
import random
import math
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

def readfile(x):
    data = []
    c = 0
    starttimeinmillis = int(round(time.time()))
    try:
        file = sys.argv[x]
    except IndexError:
        print("Improper Arguments")
        sys.exit()

    with open(file) as datafile:
        for line in datafile:
            c += 1
            if (c % 4 == 0):
            	data.append([int(l) for l in line.split()])
			

        print("time took:", int(round(time.time())) - starttimeinmillis, "seconds")
        return data


def ParseFileTrainingLabels(labelfile):	
	trainlabels = {}
	file = open(labelfile,'r')
	line = file.readline()
	while (line != ''):
		arr = line.split()
		trainlabels[int(arr[1])] = int(arr[0])
		line = file.readline()
	file.close()
	return trainlabels

def attrCalculation(data,trainlabels):
	LinearSVM=LinearSVC(C=1.0, penalty='l1', dual=False).fit(data, [x[0] for x in trainlabels])
	print("End of linearSVM")
	SVMscore = LinearSVM.coef_
	#print(len(SVMscore[0]))
	print("Coefficient End")
	attr = []
	attr_cols = []
	#count=0
	for l in range(len(SVMscore[0])):
		print(l)
		if (SVMscore[0][l] != 0.0):
			#print("if")
			#count+=1
			attr_cols.append(int(l))
			rowdata = []
			for i in range(len(data)):
				rowdata.append(data[i][l])
			attr.append(rowdata)
		bestattrs_data = [list(map(float, x)) for x in zip(*attr)]

	#bestfeaturesdata_file = open('bestattrs_data', 'w')
	#print(count)
	#print("Number of attr selected:", len(bestattrs_data[0]))

	#for i in range(0, len(bestattrs_data), 1):
	#	for j in range(0, len(bestattrs_data[i]), 1):
	#		bestfeaturesdata_file.write(str(int(bestattrs_data[i][j])) + " ")
	#	bestfeaturesdata_file.write('\n')

	#bestfeaturesdata_file.close()

	return bestattrs_data, attr_cols

def createTrainlabel(data, labels):
	org_data = [x for x in data]
	org_labels = [x for x in labels]
	#print("Original data:", len(org_data))
	rows = len(data)
	#print(int(rows * 0.1))

	print("starting...")
	predictdata = []
	actualLabels=[]
	rangef = int(rows * 0.1)
	predictlabels = []
	if rangef < 1:
		rangef = 1
	for x in range(0, rangef, 1):
		randomnum = random.randrange(0, (len(data) - 1))
		predictlabels.append(randomnum)
		predictdata.append(data[randomnum])
		actualLabels.append(labels[randomnum][0])
		del data[randomnum]
		del labels[randomnum]

	#print(actualLabels)
	return org_data, org_labels, data, labels, predictdata, predictlabels, actualLabels


def predictLabels(traindata, trainlabels, testdata, predictlabels):

    clf = LinearSVC(max_iter=15000000, tol=0.00000001).fit(traindata, [x[0] for x in trainlabels])
    # clf = SVC(C=10, kernel='linear', verbose=1, tol=0.00000001).fit(data,[x[0] for x in labels])
    # clf = tree.DecisionTreeClassifier().fit(X,[x[0] for x in y])

    predictedlabels = clf.predict((testdata))
    # predictedl=clf.predict(scale(predict))     #For LinearSVC

    #predictedlabels_file = open('predictedlabels', 'w')

    #for i in range(0, len(testdata), 1):
        #print(str(predictedlabels[i])+" "+str(i))
    #predictedlabels_file.write(str(predictedlabels[i]) + " " + str(predictlabels[i]) + '\n')

    #predictedlabels_file.close()
    return True,predictedlabels

def CalAccuracy(actualLabels,predictedlabels):

	print(actualLabels,len(actualLabels))
	print(predictedlabels,len(predictedlabels))

	fn = 0
	fp = 0
	tp = 0
	tn = 0

	for i in predictedlabels:
		if (actualLabels[i] == 0):
			if (predictedlabels[i] == 0):
				fn += 1
			else:
				fp += 1

		else:
			if (predictedlabels[i] == 1 and actualLabels[i] == 1):
				tp += 1
			else:
				tn += 1


	print(fn)
	print(fp)
	print(tp)
	print(tn)

	balance_error1=0;
	balance_error2=0;

	if((int(fp) + int(fn))!=0):
		balance_error1 = float(fp) / (int(fp) + int(fn))

	if((int(tn) + int(tp))!=0):
		balance_error2 = float(tn) / (int(tn) + int(tp))
	
	balance_error = (balance_error1 + balance_error2) / 2

	accuracy = ((fn + tp) / len(predictedlabels)) * 100
	
	return accuracy
		
def main():
	startTime = time.time()
	datafile = sys.argv[1]	
	labelfile = sys.argv[2]
	#data = ParseFileDataPoints(datafile)
	data =readfile(1);

	datarows = len(data)
	datacols = len(data[0])
	print(datarows)
	print(datacols)
	labels=readfile(2)
	#trainlabels = ParseFileTrainingLabels(labelfile)
	print(len(labels))


	print("Feature START")
	bestfeatures_data, feature_cols = attrCalculation(data, labels)
	#print(len(bestfeatures_data))
	#print(len(feature_cols))
	print("Feature END")

	print("Predicted Labels START")
	org_data, org_labels, traindata, trainlabels, predictdata, predictlabels, actualLabels = createTrainlabel(bestfeatures_data,labels)
	isPredicted,predictedlabels = predictLabels(traindata, trainlabels, predictdata, predictlabels)
	accuracy=CalAccuracy(actualLabels,predictedlabels)
	print("Accuracy: ",accuracy)
	print("Predicted Labels END")

	#org_data, org_labels, data, labels, predictdata, predictlabels, actualLabels = createTrainlabel(org_data, org_labels)
	#isPredicted,predictedlabels = predictLabels(data, labels, predictdata, predictlabels)
	#accuracy=CalAccuracy(actualLabels,predictedlabels)
	#print("Accuracy: ",accuracy)

	totalTime = (time.time() - startTime)
	print("Total Execution Time: ", int(totalTime / 60), "minutes", int(totalTime % 60), "seconds")

main()