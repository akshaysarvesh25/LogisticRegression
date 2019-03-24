import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from operator import itemgetter
import matplotlib.pyplot as plt


class DataGet:
    def __init__(self,sample_train,sample_test):
        self.x_train_path = sample_train
        self.x_test_path  = sample_test

    def GetSamples(self):
        x_train = np.genfromtxt(self.x_train_path,delimiter=',')
        x_train = np.array(x_train)
        x_test = np.genfromtxt(self.x_test_path,delimiter=',')
        x_test = np.array(x_test)
        return x_train,x_test

    def CollateData(self):
        x_train,x_test = self.GetSamples()
        y_train =  x_train[:,(x_train.shape[1] - 1)]
        y_test = x_test[:,(x_test.shape[1] - 1)]
        x_train_ip = x_train[:,0:(x_train.shape[1] - 1)]
        x_test_ip  = x_test[:,0:(x_test.shape[1] - 1)]
        y_test = np.array(y_test).reshape(-1,1)
        y_train = np.array(y_train).reshape(-1,1)
        new_col = x_test_ip.sum(1)[...,None]
        x_test_ip = np.hstack((x_test_ip, new_col))
        return x_train_ip,y_train,x_test_ip,y_test


def RandomizedTrainingDataGenerator(x_train_ip,y_train_ip,NumbSamples):
    Samples_pos1 = np.array(random.sample(range(0, len(x_train_ip)/2), NumbSamples))
    Samples_neg1 = np.array(random.sample(range(len(x_train_ip)/2,len(x_train_ip)), NumbSamples))
    Samples_pos1 = np.hstack((Samples_pos1))#,Samples_neg1))
    Samples_neg1 = np.hstack((Samples_neg1))
    Samples = np.hstack((Samples_pos1,Samples_neg1))
    x_samples = np.array(itemgetter(*Samples)(x_train_ip))
    new_col = x_samples.sum(1)[...,None]
    x_samples = np.hstack((x_samples, new_col))
    return x_samples,np.array(itemgetter(*Samples)(y_train_ip))

def GeneratePredictor(x_train_rand1,y_train_rand1):
    return LogisticRegression().fit(x_train_rand1, y_train_rand1)

def PredictOutput(reg1,x_test1):
    y_predicted = reg1.predict(x_test1)
    return y_predicted

def OutputError(y_predict1,y_test1):
    y_predict1 = np.array(y_predict1).reshape(-1,1)
    y_test1 = np.array(y_test1).reshape(-1,1)
    return (y_predict1 != y_test1).sum()


def main():
    InputPath_trainingSet = '../Normal_train_10D.txt'
    InputPath_testingSet = '../Normal_test_10D.txt'
    DataObj = DataGet(InputPath_trainingSet,InputPath_testingSet)

    x_train_ip,y_train,x_test_ip,y_test = DataObj.CollateData()

    Train_data_numbers = [20, 50, 75, 100,250, 500]
    error =  np.zeros(len(Train_data_numbers))
    error_ = []
    iterations = 1000

    for iters in range(1,iterations):

        for Train_data_numbers_ in Train_data_numbers:

            x_train_rand,y_train_rand = RandomizedTrainingDataGenerator(x_train_ip,y_train,Train_data_numbers_)
            reg = GeneratePredictor(x_train_rand,y_train_rand)
            y_predict = PredictOutput(reg,x_test_ip)
            error_.append(OutputError(y_predict,y_test))


        error = np.vstack((error,error_))
        error_ = []

    error = np.array(error)
    error = np.sum(error, axis=0)
    error = error/iterations

    plot1,= plt.plot(Train_data_numbers,error,'b.',label='Logistic regression error')
    plt.grid()
    plt.xlabel('Samples')
    plt.ylabel('Training error')
    plt.title('Training error for Logistic Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
