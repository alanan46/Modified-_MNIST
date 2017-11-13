#!/usr/bin/env python
import numpy as np
class aNN:

    """ input_x are list of preprocessed features,
        say x[0]=[0.5,0.7,0.8] is the 3 feature values of the first image
     could be adjusted, epochs means how many steps do we run for convergence
     In this design, we do cross Validation outside the class
     i.e. we feed in the input_x,input_y as they would have been seperated already
     """
    #  #for onehot encoding
    label_class=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
     11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25,
     27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
    #this is our one hot encoding look up array, it is same mathematically as a I_40, Identity matrix 40x40


    one_hot_array=np.array([[1 if j==i else 0 for j in xrange(40)] for i in xrange(40) ])


    def __init__(self,input_x,input_y,num_of_hidden_layer,num_of_neuron_per_layer,learn_rate,epochs):
        self.num_of_hidden_layer=num_of_hidden_layer
        self.num_of_neuron_per_layer=num_of_neuron_per_layer
        #input_x needs to be numpy array
        self.input_x=input_x
        self.input_y=input_y
        #number of features
        self.num_of_ft=len(self.input_x[0])
        self.learn_rate=learn_rate
        self.epochs=epochs
        self.trained=False

        #in debug mode this can be turned of to avoid plateau
        np.random.seed(0)

        # -.5,.5 is the range for the weights
        #(self.num_of_neuron_per_layer,self.num_of_neuron_per_layer) is the dimension of the numpy array for weights
        #this attribute will contain the weights for one layer of neurons
        #i.e weights[0] contains the weights used to compute the first hidden layer
        #weights[0][0] contains the weights to be dotted with features x1 to xn to get the value of first neuron in hidden layer 1
        # weights [-1] contains all the weights to compute the output layer(non-hidden )
        #a helper array
        sizeArr=[]
        #init weights to calculate input_layer --> first hidden layer
        sizeArr.append( (self.num_of_neuron_per_layer , self.num_of_ft ) )
        for _ in xrange(self.num_of_hidden_layer-1):
            sizeArr.append( (self.num_of_neuron_per_layer,self.num_of_neuron_per_layer) )
        #init weights to calculate last hidden layer --> output layer
        sizeArr.append((40,self.num_of_neuron_per_layer) )

        #self.weights attribute stores the weights matrix
        #init weights matrix from -0.5 to 0.5 
        self.weights=[np.random.uniform(-.5, .5,size) for size in sizeArr]
        self.layers=[]
        self.deltas=[]



    def get_onehot_label(self,original_label):
        return aNN.one_hot_array[ aNN.label_class.index(original_label) ]

    #return relu derivative, customized to process a vector of x
    def get_relu_deriv(self,x):
        x[x<=0]=0
        x[x>0]=1
        return x

    #calculate the sigmoid function result based on input number
    def get_sigmoid(self,x):
        #an interval is set so that result is numerically stable
        x=np.clip( x, -500, 500 )
        return 1.0/(1+np.exp(-x))

    #calculate the derivative of sigmoid function result on input number
    def get_sigmoid_deriv(self,x):
        tmp=self.get_sigmoid(x)
        return tmp*(1.0-tmp)

    #utility function to calcuate the predictions
    def get_accuracy(self,predictions,labels):
        err=0.0
        print ("length :",len(predictions))
        for (prediction,label) in zip(predictions,labels):
            if (prediction !=label):
                err+=1
        print("accuracy:",1.0-err/len(predictions) )
        return (1-err/len(predictions))

    #calcuate Rectified Linear Unit we may want to consider using relu instead of sigmoid
    def get_relu(self, data_array):
        return np.maximum(data_array,0)


    # one iteration of forward pass
    #every time we update the matrix self.layers
    def forward_pass(self,layer_input):
        # clean up layers matrix
        self.layers=[]
        # append input layer
        self.layers.append(layer_input)
        i=0
        for layer_weights in self.weights:
            # use previous layer to get current layer 
            layer_input=self.layers[i]
            i+=1
            #dim layer input for first layer =1x4096 for raw pixel and dim layer_weights[0]=num_of_neuron_per_layer x 4096 for raw pixel
            layer_output=self.get_sigmoid(np.dot(layer_input,layer_weights.T))
            self.layers.append(layer_output)
        # after function finished we have a layer matrix  stored in self.layers

    # store the deltas in self.deltas
    def get_deltas(self,label):
        #delta_N+H+1 = o(1-o)(y-o)
        #o(1-o) is self.get_sigmoid_deriv(self.layers[-1])
        #empty the last deltas
        self.deltas=[]
        delta=self.get_sigmoid_deriv(self.layers[-1])*(self.get_onehot_label(label)-self.layers[-1] )
        #self.layers[-2::-1] is the layer matrix except the output_layer in reversed order
        for layer,layer_weights in zip(self.layers[-2::-1],self.weights[::-1]):
                #store it in a class attribute
                self.deltas.append(delta)
                #delta_h=o_h*(1-o_h)*weights_N+H+1 dot delta_N+H+1
                #delta dimension 1x40 for o_N+H+1 and dim layer_weights= 40 x num_neuron per layer
                delta=np.dot(delta,layer_weights)*self.get_sigmoid_deriv(layer)

    def back_prop(self,label):
        #first get deltas
        self.get_deltas(label)
        result=[]
        for layer_weights,layer,delta in zip(self.weights,self.layers,self.deltas[::-1]):
            # gradient descent
                                            #outer(a,b) will return a matrix dim:MxN given dim(a)=M dim(b)=N
            updated_weights=layer_weights+ self.learn_rate*np.outer(delta,layer)
            result.append( updated_weights )
        #store weights in class attribute 
        self.weights=result

    #this function will use the back_prop and forward_pass
    # after training, it will store the weights in a .npy file so that we can load later
    # and use it to predict
    def train(self,file_path="./weights.npy"):
        self.trained=True
        for i in xrange(self.epochs):
            print("on epoch:", i)
            # j=1
            for (x,y) in zip(self.input_x,self.input_y):
                #updates the weights matrix after every round of back_prop
                self.forward_pass(x)
                self.back_prop(np.array(y))

        #write down the valuable trained result into a file so that we can reuse this particular set of weights for refining
        #w+ means overwrite the file if exist, can be changed
        with open(file_path,'w+') as f:
            np.save(f,self.weights)


    def predict(self,prediction_x):
        if(not self.trained):self.train()
        self.forward_pass(prediction_x)
        layer=self.layers[-1]
        print('layer',layer)
        # get the most probable index of the output layer
        index=np.argmax(layer)
        print ('index in class array:',index)
        print("prediciton:", aNN.label_class[index])
        return aNN.label_class[index]
