#!/usr/bin/env python
import numpy as np
# from sklearn.model_selection import train_test_split
class NN:
    """ input_x are list of preprocessed features,
        say x[0]=[0.5,0.7,0.8] is the 3 feature values of the first image
     could be adjusted, epochs means how many steps do we run for convergence
     In this design, we do cross Validation outside the class
     i.e. we feed in the input_x,input_y as they would have been seperated already
     """
     """ we assume in current design that number of feature = number of neuron per layer """
     np.
    def __init__(self,input_x,input_y,num_of_hidden_layer,num_of_neuron_per_layer,learn_rate,epochs):
        self.num_of_hidden_layer=num_of_hidden_layer
        self.num_of_neuron_per_layer=num_of_neuron_per_layer
        #input_x needs to be numpy array
        self.input_x=input_x
        self.input_y=input_y
        self.learn_rate=learn_rate
        self.epochs=epochs
        self.trained=False
        np.random.seed(0)
        # 0,1 is the range for the weights
        #(self.num_of_neuron_per_layer,self.num_of_neuron_per_layer) is the dimension of the numpy array for weights
        #this attribute will contain the weights for one layer of neurons
        #i.e weights[0] contains the weights for the neurons of the entire layer 1, the first hidden layer
        #weights[0][0] contains the weights for features x1 to xn of the first neuron in layer 1
        # weights [-1] contains all the weights of the output layer(non-hidden )
        self.weights=[np.random.uniform(0, 1,size) for size in
        [(self.num_of_neuron_per_layer,self.num_of_neuron_per_layer)
        for _ in xrange(self.num_of_hidden_layer+1)]
        ]

    #calculate the sigmoid function result based on input number
    def get_sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    #calculate the derivative of sigmoid function result on input number
    def get_sigmoid_deriv(self,x):
        tmp=self.get_sigmoid(x)
        return tmp*(1.0-tmp)

    #utility function to calcuate the predictions
    def get_accuracy(self,predictions,labels):
        err=0.0
        for (prediction,label) in (predictions,labels):
            if (prediction !=label):err+=1
        return (1-err/len(predictions))

    #calcuate Rectified Linear Unit we may want to consider using relu instead of sigmoid
    def get_relu(self, data_array):
        return np.maximum(data_array,0)


    # one iteration of forward pass
    def forward_pass(self,layer_input):
        # for the 0 layer,aka the input layer, just yield the data, i.e the output for layer 0
        yield layer_input
        #weights is an array of weights
        for layer_weights in self.weights:
            # this will yield o_i, output for layer i
            yield get_sigmoid(np.dot(layer_input,layer_weights.T))


    # evaluation function to calcuate the difference of the prediction and the true label
    # empty for now
    # should return a score that we wish to minimize
    def eval_output(self,label,output_layer):



    def get_delta(self,label,layers):
        tmp=eval_output(label,layers[-1])
        #tmp is our delta_N+H+1
        tmp*=get_sigmoid_deriv(layers[-1])
        #starting from the first hidden layer aka layers[-2] with the last elements of weights
        for layer,layer_weights in (layers[-2::-1],self.weights[::-1]):
            #yield it and calcuate the next delta after yielding
            yield tmp
            tmp=np.dot(tmp,layer_weights.T)*get_sigmoid_deriv(layer)
    def back_prop(self,label,layers):
        #store the deltas in a list
        delta_array=reversed(list(self.get_delta(label,layers)))
        result=[]
        for layer_weights,layer,delta in (self.weights,layers,delta_array):
            # gradient descent
            result.append(layer_weights+self.learn_rate*np.outer(layer,delta))
        return result

    #this function will use the back_prop and forward_pass
    def train(self):
        self.trained=True
        for _ in xrange(self.epochs):
            for (x,y) in (self.input_x,self.input_y):
                self.weights=self.back_prop(np.array(y),list(self.forward_pass(x)))

    def predict(self,prediction_x):
        if(not self.trained):self.train()
        for layer in self.foward_pass(prediction_x):pass
        return layer
