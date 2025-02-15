#!/usr/bin/env python

"""this is a toy example to verify the implementation of a neural network"""
import numpy as np
class aNN:

    """ input_x are list of preprocessed features,
        say x[0]=[0.5,0.7,0.8] is the 3 feature values of the first image
     could be adjusted, epochs means how many steps do we run for convergence
     In this design, we do cross Validation outside the class
     i.e. we feed in the input_x,input_y as they would have been seperated already
     """
    #  #for onehot encoding
    # label_class=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #  11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25,
    #  27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
    # this is our one hot encoding look up array, it is same mathematically as a I_40, Identity matrix 40x40
    label_class=[1,2,3]

    one_hot_array=np.array([[1 if j==i else 0 for j in xrange(3)] for i in xrange(3) ])


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
        sizeArr.append((3,self.num_of_neuron_per_layer) )

        #self.weights attribute stores the weights matrix

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
        # for the 0 layer,aka the input layer, just yield the data, i.e the output for layer 0
        self.layers=[]
        print ("linput",layer_input)
        self.layers.append(layer_input)

        #weights is an array of weights
        i=0
        for layer_weights in self.weights:
            # this will yield o_i, output for layer i
            layer_input=self.layers[i]

            i+=1
            #dim layer input for first layer =1x4096 for raw pixel and dim layer_weights[0]=num_of_neuron_per_layer x 4096 for raw pixel
            layer_output=self.get_sigmoid(np.dot(layer_input,layer_weights.T))
            print ("inforloop",layer_output)
            self.layers.append(layer_output)

    # ------this function is disabled for now------
    # should return a score that we wish to minimize in training mode
    # return a class in prediciton mode
    def eval_output(self,label,output_layer,prediction_mode=False):

        # the one_ hot function will do the dirty work for us to cast a possible label to 40 classes
        # onehot_label=tf.one_hot(indices=tf.cast(label[0], tf.int32), depth=40)
        # feed the class number to onehot and get the onhot representation
        if(not prediction_mode):
            index=label[0]
            print ("index is ",index)
            onehot_label=tf.one_hot(indices=tf.cast(index, tf.int32 ), depth=3)
        # print ('out_layer ',output_layer)
        #the output_layer will contain 40 neurons,convert them to tensor object
        #so that we can use the tf.losses.softmax_cross_entropy()
        logits=tf.convert_to_tensor(output_layer)
        predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # logit is an array, the argmax function will return the most promising class
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        # if we are in prediction mode instead of training mode, we just wish to return the class
        if(prediction_mode): return (self.sess.run(predictions['classes']),self.sess.run(predictions['probabilities']))

        # we calculate the loss using softmax_cross_entropy
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_label, logits=logits)
        res=self.sess.run(loss)
        print("loss:   ")
        print(res)
        return res


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
        print self.deltas
    def back_prop(self,label):
        #store the deltas in a list
        self.get_deltas(label)
        result=[]
        for layer_weights,layer,delta in zip(self.weights,self.layers,self.deltas[::-1]):
            # gradient descent
                                            #outer(a,b) will return a matrix dim:MxN given dim(a)=M dim(b)=N
            updated_weights=layer_weights+ self.learn_rate*np.outer(delta,layer)
            result.append( updated_weights )

        return result

    #this function will use the back_prop and forward_pass
    def train(self):
        self.trained=True
        for i in xrange(self.epochs):
            print("on epoch:", i)
            # j=1
            for (x,y) in zip(self.input_x,self.input_y):
                #updates the weights matrix after every round of back_prop
                #generate self.layers for current data pair
                # print (x,y)
                self.forward_pass(x)
                self.weights=self.back_prop(np.array(y))
                # print (self.weights)
                # print("on epoch:", i)
                # print ("on train_data : ",j)
                # j+=1
                # print ('weights: ',self.weights)

        #write down the valuable trained result into a file so that we can reuse this particular set of weights for refining
        #or train another data set
        #w+ means overwrite the file if exist, can be changed
        # with open("./weights.npy",'w+') as f:
        #     np.save(f,self.weights)

####!!!!!! in debug mode
    def predict(self,prediction_x):
        if(not self.trained):self.train()
        self.forward_pass(prediction_x)
        for layer in self.layers :pass

        print('layer',layer)
        index=np.argmax(layer)
        print ('index in class array:',index)
        print("prediciton:", aNN.label_class[index])
        return aNN.label_class[index]
