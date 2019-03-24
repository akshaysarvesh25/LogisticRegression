import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from operator import itemgetter
import matplotlib.pyplot as plt


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
    #y_predicted = [-1 if (x<=0) else 1 for x in y_predicted]
    return y_predicted

def OutputError(y_predict1,y_test1):
    y_predict1 = np.array(y_predict1).reshape(-1,1)
    y_test1 = np.array(y_test1).reshape(-1,1)
    return (y_predict1 != y_test1).sum()



x_train = np.genfromtxt('../Uniform_train.txt',delimiter=',')
x_train = np.array(x_train)

x_test = np.genfromtxt('../Uniform_test.txt',delimiter=',')
x_test = np.array(x_test)



y_train =  [sub_list[2] for sub_list in x_train]
x_train_1 = [[sub_list[0]] for sub_list in x_train]
x_train_2 = [[sub_list[1]] for sub_list in x_train]

y_test = [sub_list[2] for sub_list in x_test]
x_test_1 = [[sub_list[0]] for sub_list in x_test]
x_test_2 = [[sub_list[1]] for sub_list in x_test]


x_train_ip = np.concatenate((x_train_1,x_train_2),axis=1)

x_test_ip  = np.concatenate((x_test_1,x_test_2),axis=1)
new_col = x_test_ip.sum(1)[...,None]
x_test_ip = np.hstack((x_test_ip, new_col))



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

    #print(error_)
    #error = error_
    error = np.vstack((error,error_))



    error_ = []

error = np.array(error)
#print(error[100-2])
error = np.sum(error, axis=0)
error = error/iterations

plot1,= plt.plot(Train_data_numbers,error,'b.',label='Logistic regression error')
plt.grid()
plt.xlabel('Samples')
plt.ylabel('Training error')
plt.title('Training error for Logistic Regression')
plt.legend()
plt.show()
