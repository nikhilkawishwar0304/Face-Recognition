import cv2
import time
import os
import dlib
import openface
import pickle
import numpy as np
import pandas as pd
from sklearn.mixture import GMM
from sklearn.svm import SVC
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder

# Prints the array to 2 decimal places
np.set_printoptions(precision=2)

# Setting the variables to the paths of various directory

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
predictor_model = "shape_predictor_68_face_landmarks.dat"

def get_rap():

    # Starting the timer for image pre-processing
    start = time.time()

    # Capturing the video from the web camera
    #stream = cv2.VideoCapture(0)

    # Reading the image from the captured video
    #ret, bgrImg = stream.read()
    bgrImg = cv2.imread('Image4.jpg')

    # If the image isn't captured; captures exception
    if bgrImg is None:
        raise Exception("Unable to load image")

    # Converting the image to RGB matrix
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    
    print("  + Original size: {}".format(rgbImg.shape))
    print("Loading the image took {} seconds.".format(time.time() - start))

    # Resetting the timer for detection process
    start = time.time()

    # Create a HOG face detector
    detector = dlib.get_frontal_face_detector()

    # Create a pose predictor object
    # It takes in an image region containing some object and outputs a set of point locations that define the pose
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    face_aligner = openface.AlignDlib(predictor_model)

    # Run the HOG face detector on the image data
    detected_faces = detector(rgbImg, 1)

    print("Found {} faces in the image.".format(len(detected_faces)))

    # Loop through each face we found in the image
    itera = 0
    reps = []
    for face_rect in detected_faces:
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        itera = itera + 1

        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(itera, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(rgbImg, face_rect)

        # Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(96, rgbImg, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            raise Exception("Unable to align image.")

        # Creating a neural network with Default network model and parameters
        net = openface.TorchNeuralNet(model='nn4.small2.v1.t7', imgDim=96, cuda=False)

        # Execute the Neural Network in forward direction
        rep = net.forward(alignedFace)

        #  Appending the features for all the faces in the image
        reps.append((face_rect.center().x, rep))

    sortreps = sorted(reps, key=lambda x: x[0])
    return sortreps

def infer():
    # Open the trained classifier file and load the same
    with open('classifier.pkl','r') as f:
        (le,clf) = pickle.load(f)

    reps = get_rap()

    # If more than one faces in image then the images will be recognized left to right
    if len(reps) > 1:
        print("List of faces in image from left to right")
    for r in reps:
        rep = r[1].reshape(1, -1)
        bbx = r[0]
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]

        print("Prediction took {} seconds.".format(time.time() - start))
        print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))

def train():

    # Loading 128 Feature set and labels (called Embeddings as Image is converted into a series of numbers)
    print "Loading Embeddings"

    # Loading the existing labels file and take the first column
    fname = "/Users/knikhil/openface/generated-embeddings/labels.csv"
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]

    # Column is a path to the folder with the name of person; thus to get the name split it
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory

    # Loading the existing feature matrix from the csv file
    fname = "/Users/knikhil/openface/generated-embeddings/reps.csv"
    embeddings = pd.read_csv(fname, header=None).as_matrix()

    # Assigns the appropriate label to each image from the labels file
    le = LabelEncoder().fit(labels)

    # Transforms the label name to a number (Unique Identifier for each user name)
    labelsNum = le.transform(labels)

    # Number of classes for which the classifier is trained
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    # Builds an SVM classifier with linear kernel
    clf = SVC(C=1, kernel='linear', probability=True)

    # Fits the model on the data
    clf.fit(embeddings, labelsNum)

    # Export the classifier to a pkl file
    fName = "classifier.pkl"
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

infer()
