import sys
from sys import argv
import random
import math
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

def ParseFile(argumentNo):
    data = []
    c = 0
    readStartTime = int(round(time.time()))
    try:
        file = sys.argv[argumentNo]
    except IndexError:
        print("Improper Arguments")
        sys.exit()

    with open(file) as dFile:
        for line in dFile:
            c += 1
            if (c % 10 == 0):
            	data.append([int(l) for l in line.split()])
			

    print("time taken:", int(round(time.time())) - readStartTime)
    return data


def attrCalculation(data,trainlabels):
	LinearSVM=LinearSVC(C=1.0, penalty='l1', dual=False).fit(data, [x[0] for x in trainlabels])
	print("End of linearSVM")
	SVMscore = LinearSVM.coef_
	attr = []
	attr_cols = []
	for l in range(len(SVMscore[0])):
		#print(l)
		if (SVMscore[0][l] != 0.0):
			attr_cols.append(int(l))
			rowdata = []
			for i in range(len(data)):
				rowdata.append(data[i][l])
			attr.append(rowdata)
		bestattrs_data = [list(map(float, x)) for x in zip(*attr)]

	if(len(bestattrs_data)==0):
		return data,attr_cols

	return bestattrs_data, attr_cols

def predictLabels(traindata, trainlabels, testdata):

    clf = LinearSVC(max_iter=15000000, tol=0.00000001).fit(traindata, [x[0] for x in trainlabels])
    
    predictedlabels = clf.predict((testdata))
    
    predictedlabels_file = open('predictedlabels', 'w')

    for i in range(0, len(testdata), 1):
        predictedlabels_file.write(str(predictedlabels[i]) + " " + str(i) + '\n')

    predictedlabels_file.close()
    print("\nPredictions are stored in 'predictedlabels' file")
    return True,predictedlabels

def TestDataAttr(testdata,feature_cols):
	bestTestDataAttr=[]
	totalAttr=len(feature_cols)

	for i in range(len(testdata)):
		col=[]
		k = 0
		for j in range(len(testdata[0])):
			if(k<totalAttr and feature_cols[k]==j):
				col.append(testdata[i][j])
				k+=1
		bestTestDataAttr.append(col)

	print(k)
	best_attr_testdata_file=open('bestfeaturestestdata', 'w')
	#print("Number of Features selected in testdata:", len(bestTestDataAttr[0]))

	for i in range(0,len(bestTestDataAttr),1):
		for j in range(0, len(bestTestDataAttr[i]), 1):
			best_attr_testdata_file.write(str(bestTestDataAttr[i][j]) + " ")
		best_attr_testdata_file.write('\n')

	best_attr_testdata_file.close()
	return bestTestDataAttr

		
def main():
	startTime = time.time()
	data =ParseFile(1)
	labels=ParseFile(2)
	testdata=ParseFile(3)

	############################################
	#Feature Selection
	############################################
	print("Feature START")
	bestfeatures_data, feature_cols = attrCalculation(data, labels)
	print("Feature END")
	
	############################################
	#Extracting Attributes for test data
	############################################
	bestfeature_testdata = TestDataAttr(testdata, feature_cols)

	print("Predicted Labels START")
	isPredicted,predictedlabels = predictLabels(bestfeatures_data, labels, bestfeature_testdata)

	if (isPredicted):
		print("##Data classified successfully and stored in 'predictedlabels' file     ##")
        
	totalTime = (time.time() - startTime)
	print("Execution Time of program: ", int(totalTime / 60), "minutes", int(totalTime % 60), "seconds")

main()
