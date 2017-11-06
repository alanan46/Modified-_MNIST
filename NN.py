#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import copy
# from sklearn.model_selection import train_test_split
class NN:

    """ input_x are list of preprocessed features,
        say x[0]=[0.5,0.7,0.8] is the 3 feature values of the first image
     could be adjusted, epochs means how many steps do we run for convergence
     In this design, we do cross Validation outside the class
     i.e. we feed in the input_x,input_y as they would have been seperated already
     """
    label_class=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25,
      27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

    def __init__(self,input_x,input_y,num_of_hidden_layer,num_of_neuron_per_layer,learn_rate,epochs):
        self.num_of_hidden_layer=num_of_hidden_layer
        self.num_of_neuron_per_layer=num_of_neuron_per_layer
        #input_x needs to be numpy array
        self.input_x=input_x
        self.input_y=input_y
        self.num_of_ft=len(self.input_x[0])
        self.learn_rate=learn_rate
        self.epochs=epochs
        self.trained=False

        #in debug mode this can be turned of to avoid plateau
        # np.random.seed(0)
        #open up a tf session so that the tensor obj can be evaluated
        self.sess=tf.Session()

        # 0,1 is the range for the weights
        #(self.num_of_neuron_per_layer,self.num_of_neuron_per_layer) is the dimension of the numpy array for weights
        #this attribute will contain the weights for one layer of neurons
        #i.e weights[0] contains the weights used to compute the first hidden layer
        #weights[0][0] contains the weights to be dotted with features x1 to xn to get the value of first neuron in hidden layer 1
        # weights [-1] contains all the weights to compute the output layer(non-hidden )
        sizeArr=[]
        #init weights to calculate input_layer --> first hidden layer
        sizeArr.append( (self.num_of_neuron_per_layer , self.num_of_ft ) )
        for _ in xrange(self.num_of_hidden_layer-1):
            sizeArr.append( (self.num_of_neuron_per_layer,self.num_of_neuron_per_layer) )
        #init weights to calculate last hidden layer --> output layer
        sizeArr.append((40,self.num_of_neuron_per_layer) )
        self.weights=[np.random.uniform(-.5, .5,size) for size in sizeArr]

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

            layer_input=self.get_sigmoid(np.dot(layer_input,layer_weights.T))
            yield layer_input
    # evaluation function to calcuate the difference of the prediction and the true label
    # empty for now
    # should return a score that we wish to minimize in training mode

    # return a class in prediciton mode
    def eval_output(self,label,output_layer,prediction_mode=False):

        # the one_ hot function will do the dirty work for us to cast a possible label to 40 classes
        # onehot_label=tf.one_hot(indices=tf.cast(label[0], tf.int32), depth=40)
        # feed the class number to onehot and get the onhot representation
        if(not prediction_mode):
            index=NN.label_class.index(label[0])
            print ("index is ",index)
            onehot_label=tf.one_hot(indices=tf.cast(index, tf.int32 ), depth=40)
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
        if(prediction_mode): return (self.sess.run( predictions['classes'] ), self.sess.run( predictions['probabilities']))

        # we calculate the loss using softmax_cross_entropy
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_label, logits=logits)
        res=self.sess.run(loss)
        print("loss:   ")
        print(res)
        return res


    def get_delta(self,label,layers):
        self.eval_output(label,layers[-1])
        index=NN.label_class.index(label[0])
        onehot_label=tf.one_hot(indices=tf.cast(index, tf.int32 ), depth=40)
        # tmp=-1*self.eval_output(label,layers[-1])
        tmp=(self.sess.run(onehot_label)-layers[-1])
        #tmp is our delta_N+H+1
        tmp*=self.get_sigmoid_deriv(layers[-1])
        # hidden_layers=copy.deepcopy(layers)
        # del hidden_layers[0] # delete the input layer
        #starting from the last hidden layer aka layers[-2] with the last elements of weights
        for layer,layer_weights in zip(layers[-2::-1],self.weights[::-1]):
            #yield it and calcuate the next delta after yielding

                yield tmp
                tmp=np.dot(tmp,layer_weights)*self.get_sigmoid_deriv(layer)

    def back_prop(self,label,layers):
        #store the deltas in a list
        delta_array= list(self.get_delta(label,layers))
        delta_array.reverse()
        result=[]
        for layer_weights,layer,delta in zip(self.weights,layers,delta_array):
            # gradient descent
            updated_weights=layer_weights+ self.learn_rate*np.outer(delta,layer)
            result.append( updated_weights )
        return result

    #this function will use the back_prop and forward_pass
    def train(self):
        self.trained=True
        for i in xrange(self.epochs):

            j=1
            for (x,y) in zip(self.input_x,self.input_y):
                #updates the weights matrix after every round of back_prop
                self.weights=self.back_prop(np.array(y),list(self.forward_pass(x)))
                print("on epoch:", i)
                print ("on train_data : ",j)

                j+=1
                # print ('weights: ',self.weights)

        #write down the valuable trained result into a file so that we can reuse this particular set of weights for refining
        #or train another data set
        #w+ means overwrite the file if exist, can be changed
        # with open("./weights.npy",'w+') as f:
        #     np.save(f,self.weights)

####!!!!!! in debug mode 
    def predict(self,prediction_x):
        if(not self.trained):self.train()
        for layer in self.forward_pass(prediction_x):pass
        #label is dummy here, the value we give is not used  since we are in prediction mode
        index,prob=self.eval_output(label=0,output_layer=layer,prediction_mode=True)
        print prob
        return NN.label_class[index]
