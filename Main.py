import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Loads the cascade

ds_train = None
ds_validate = None
batch_size = 32
class_names = []
for folder in sorted(os.listdir("Test Images/Cropped")):
    class_names.append(folder)


# Defines the architecture for the model we are using ---------------------------------------------------------------------------------------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu))  # Creates the first neuron layer with 128 neurons and the activation 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)) # function of leaky relu which is the standard 
model.add(tf.keras.layers.Dense(len(class_names), activation="sigmoid")) # Output with 10 possible values 0-9 with a softmax activation as its a probability distribution

def compilemodel():
    global model
    # Defines the paramaters for the training of the model -------------------------------------------------------------------------------
    model.compile(optimizer = 'adam',                       # How the training is optimised, adam is a good default but there are many others
              loss ='categorical_crossentropy',  # Degree of error allowed in the model
              metrics=['accuracy'])

def loadingdataset():
    global ds_train 
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
            'Test Images/Done',
            labels ='inferred',         # Labels are inferred from the directory structure on the disk
            label_mode = 'categorical', # Encoding method of label; categorical = cat, dog |int = 0, 1, 2 |binary = true, false
            color_mode = 'rgb',         # Colour of images, can be either grayscale or rgb
            batch_size = batch_size,    # Size of batches of data, default is 32
            image_size = (256, 256),    # Size of an image inside the dataset, will auto resize if not this size, 100% needed!!!!!
            shuffle = True,             # Will shuffle the dataset into a random order each epoch to get better results
            seed = 123,                 # Allows others to get the same result by putting in a similar seed
            validation_split = 0.1,     # Tells you what percent of the data is going to be in the validation set, to test the model
            subset = 'training',        # Tells the dataset what type it is, validation or training
            )
    global ds_validate
    ds_validate = tf.keras.preprocessing.image_dataset_from_directory(
            'Test Images/Done',
            labels ='inferred',          # Labels are inferred from the directory structure on the disk
            label_mode = 'categorical',  # Encoding method of label; categorical = cat, dog |int = 0, 1, 2 |binary = true, false
            color_mode = 'rgb',          # Colour of images, can be either grayscale or rgb
            batch_size = batch_size,    # Size of batches of data, default is 32
            image_size = (256, 256),    # Size of an image inside the dataset, will auto resize if not this size, 100% needed!!!!!
            shuffle = True,             # Will shuffle the dataset into a random order each epoch to get better results
            seed = 123,                 # Allows others to get the same result by putting in a similar seed
            validation_split = 0.1,      # Tells you what percent of the data is going to be in the validation set, to test the model
            subset = 'validation',       # Tells the dataset what type it is, validation or training
            )

def trainmodel():
    global model
    print("How many epochs to run for?")
    runs = int(input())

    # Model training time baby
    model.fit(ds_train, validation_data=ds_validate, epochs=runs)   # Takes the too training values and trains for the number of epochs where 
                                            #1 epoch = 1 full run through of the data set

    val_loss, val_acc = model.evaluate(ds_validate)
    print(val_loss, val_acc)
    menu()

def savemodel():
    model.save('numreader.model') # Saves a model
    menu()

def loadmodel():
    global model
    model = tf.keras.models.load_model('numreader.model') # Loads a model
    model.summary()
    #loss, acc = model.evaluate(, test_labels, verbose=2)
    #print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    menu()

def predict():
    capture = cv2.VideoCapture('/dev/v4l/by-id/usb-OmniVision_Technologies__Inc._USB_Camera-B4.09.24.1-video-index0')
    
    while True:
        # test = cv2.imread('Test Images/Test.jpg')
        ret, test = capture.read()
        # gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)    # Convert to grayscale
        # gray = cv2.resize(gray, None, fx=0.95, fy=0.95)
        faces = face_cascade.detectMultiScale(test, 1.10, 10)
        for (x, y, w, h) in faces:       
            roi = test[y:y+h, x:x+w]
            roi = cv2.resize(roi,(256,256)) /255
            roi = np.expand_dims(roi, axis=0)
            # roi = roi.reshape(-1, 256 ,256 , 3)
            y_pred = model.predict(roi)
            cv2.rectangle(test, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(test, str(class_names[y_pred.argmax()]), (x - 20, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Test", test)
        cv2.createButton("Incorrect Prediction" ,incorrectprediction  ,None ,cv2.QT_PUSH_BUTTON, 1)

    # Get prediction and visualize
    # for i in range(10, 16):
        # plt.subplot(280 + (i%10+1))
        # plt.imshow(x_validate[i].reshape(28, 28), cmap=plt.cm.gray)
        # plt.title(x_validate[i].argmax())
    # plt.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    closewindow()

def closewindow():
    cv2.destroyAllWindows()
    menu()

def incorrectprediction():
    print("AHHHH")
    closewindow()

def menu():
    print("--------------------------------")
    print("1 - train model                 ")
    print("2 - save model                  ")
    print("3 - load model                  ")
    print("4 - test model                  ")
    print("5 - quit                        ")
    print("--------------------------------")
    try:
        choice = int(input())
    except:
        menu()
    if choice == 1:
        trainmodel()
    elif choice == 2:
        savemodel()
    elif choice == 3:
        loadmodel()
    elif choice == 4:
        predict()
    elif choice == 5:
        quit()
    else:
        print("Invalid input")
        menu()


compilemodel()
loadingdataset()
menu()
