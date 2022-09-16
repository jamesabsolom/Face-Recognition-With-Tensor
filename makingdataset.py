import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

ds_train = None
ds_validate = None
batchsize = 2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Defines the architecture for the model we are using ---------------------------------------------------------------------------------------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) # Input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))   # Creates the first neuron layer with 128 neurons and the activation function 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))   # of relu which is the standard 
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # Output with 10 possible values 0-9 with a softmax activation as its a 
                                                               # probability distribution

# Defines the paramaters for the training of the model --------------------------------------------------------------------------------------
model.compile(optimizer = 'adam',                       # How the training is optimised, adam is a good default but there are many others
              loss ='sparse_categorical_crossentropy',  # Degree of error allowed in the model
              metrics=['accuracy'])

def loadingdataset():
    global
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
            'path',
            labels ='inferred'          # Labels are inferred from the directory structure on the disk
            label_mode = 'categorical'  # Encoding method of label; categorical = cat, dog |int = 0, 1, 2 |binary = true, false
            color_mode = 'rgb'          # Colour of images, can be either grayscale or rgb
            batch_size = batch_size,    # Size of batches of data, default is 32
            image_size = (256, 256),    # Size of an image inside the dataset, will auto resize if not this size, 100% needed!!!!!
            shuffle = True,             # Will shuffle the dataset into a random order each epoch to get better results
            seed = 123,                 # Allows others to get the same result by putting in a similar seed
            validation_split = 0.1      # Tells you what percent of the data is going to be in the validation set, to test the model
            subset = 'training'         # Tells the dataset what type it is, validation or training
            )
    global
        ds_train = tf.keras.preprocessing.image_dataset_from_directory(
            'path',
            labels ='inferred'          # Labels are inferred from the directory structure on the disk
            label_mode = 'categorical'  # Encoding method of label; categorical = cat, dog |int = 0, 1, 2 |binary = true, false
            color_mode = 'rgb'          # Colour of images, can be either grayscale or rgb
            batch_size = batch_size,    # Size of batches of data, default is 32
            image_size = (256, 256),    # Size of an image inside the dataset, will auto resize if not this size, 100% needed!!!!!
            shuffle = True,             # Will shuffle the dataset into a random order each epoch to get better results
            seed = 123,                 # Allows others to get the same result by putting in a similar seed
            validation_split = 0.1      # Tells you what percent of the data is going to be in the validation set, to test the model
            subset = 'validation'       # Tells the dataset what type it is, validation or training
            )

def trainmodel():
    # Model training time baby
    model.fit(x_train, y_train, epochs=5)   # Takes the too training values and trains for the number of epochs where 1 epoch = 1 full run
                                            # through of the data set

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

def savemodel():
    model.save('numreader.model') # Saves a model
    new_model = tf.keras.models.load_model('numreader.model') # Loads a model

def predict():
    y_pred = new_model.predict(x_test)

    # Get prediction and visualize
    for i in range(10, 16):
        plt.subplot(280 + (i%10+1))
        plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray)
        plt.title(y_pred[i].argmax())
    plt.show()

    while True:
        if 0xFF == ord('q'):
            break

