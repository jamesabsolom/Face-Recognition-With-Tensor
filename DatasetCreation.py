import cv2
import os
import numpy
import imutils
from tqdm import tqdm
from rembg import remove

#capture = cv2.VideoCapture('/dev/v4l/by-id/usb-OmniVision_Technologies__Inc._USB_Camera-B4.09.24.1-video-index0')
capture = cv2.VideoCapture(2)
ret, frame = capture.read()
    #if ret == 0:
    #print("U BROKE IT DUMMY")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

name = None

if not os.path.exists("Test Images"):
    os.mkdir("Test Images")

if not os.path.exists("Test Images/Manip1"):
    os.mkdir("Test Images/Manip1")

if not os.path.exists("Test Images/Manip2"):
    os.mkdir("Test Images/Manip2")

if not os.path.exists("Test Images/Done"):
    os.mkdir("Test Images/Done")

def definename():
    print("Input name of subject")
    name = input()
    choice = ""

    folderpathcap = os.path.join("Test Images/Base", name)
    folderpathface = os.path.join("Test Images/Manip1", name)
    folderpathdone = os.path.join("Test Images/Manip2", name)
    folderpathfinalize = os.path.join("Test Images/Done", name)

    while not os.path.exists(folderpathcap):
        print(name + " does not exist in dataset, do you want to add it? y/n ")
        try:
            choice = input()
        except:
            print("Invalid input")
        if choice == "y":
            os.mkdir(folderpathcap)
            if not os.path.exists(folderpathface):
                os.mkdir(folderpathface)
            if not os.path.exists(folderpathdone):
                os.mkdir(folderpathdone)
            if not os.path.exists(folderpathfinalize):
                os.mkdir(folderpathfinalize)

        elif choice == "n":
            break

    return folderpathcap, folderpathface, folderpathdone, folderpathfinalize, name


def recordvideo(numofimg, folderpathcap):
    temp = 0
    for x in tqdm(range(numofimg)):
        temp=0
        while temp < 1:
            ret, frame = capture.read()
            if ret == 0:
                print("U BROKE IT DUMMY")
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # convert to grayscale
                faces = face_cascade.detectMultiScale(gray, 1.10, 10)
                if len(faces) > 0:
                    #print(temp)
                    cv2.imwrite(folderpathcap + "/" + (name + "_" + str(x) + ".jpg"), frame)
                    temp = temp + 1
                    cv2.imshow("Webcam Feed",frame)
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
    capture.release()
    cv2.destroyAllWindows()
    #manipulate1()
    
def manipulate1(folderpathcap, folderpathface, folderpathfinalize, name):
    print("Manipulating")
    allFaces = []
    pics = os.listdir(folderpathcap)
    for item in tqdm(pics):
        img = cv2.imread(os.path.join(folderpathcap, item))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.10, 10)
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            removed = remove(roi)
            allFaces.append(roi)
            allFaces.append(removed)
    yesnoui(allFaces, folderpathface)

def yesnoui(allFaces, folderpathface)
    temp = 0
    for item in allFaces:
        temp = temp+1
        while True:
            cv2.imshow("K - Is Face, L - Not Face",item)
            if cv2.waitKey(1) & 0xFF == ord('k'):
                cv2.imwrite((os.path.join(folderpathface, str(temp) + name + ".jpg")), item)
                break
            elif cv2.waitKey(1) & 0xFF == ord('l'):
                break

    os.rename(folderpathcap, folderpathfinalize)
    os.mkdir(folderpathcap)

def manipulate2(folderpathface, folderpathdone):
    print("Final Manipulation")
    pics = os.listdir(folderpathface)
    for item in tqdm(pics):
        img = cv2.imread(os.path.join(folderpathface, item))
        
        #img = remove(img)

        temp2 = img;
        cv2.imwrite((folderpathdone + "/Regular_" + item), temp2)
        
        temp3 = cv2.flip(img ,1)
        cv2.imwrite((folderpathdone + "/Flipped_" + item), temp3)
        
        temp4 = cv2.resize(img, None, fx = 0.8, fy = 0.8)
        cv2.imwrite((folderpathdone + "/Zoomed_" + item), temp4)

        temp5 = imutils.rotate(img, 45)
        cv2.imwrite((folderpathdone + "/Rotate45_" + item), temp5)

        temp6 = imutils.rotate(img, -45)
        cv2.imwrite((folderpathdone + "/Rotate-45_" + item), temp6)

        temp7 = cv2.GaussianBlur(img, (0, 0), 3)
        cv2.imwrite((folderpathdone + "/Blur_" + item), temp7)

        temp8 = cv2.addWeighted(img, 0.5, img, 0.5, 0)
        cv2.imwrite((folderpathdone + "/Sharp_" + item), temp8)

        os.remove(os.path.join(folderpathface, item))

#recordvideo()
#manipulate1()
#manipulate2()
#print("Done")


