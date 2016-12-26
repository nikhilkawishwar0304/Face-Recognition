FACE RECOGNITION

As a human, our brain is wired to recognize faces automatically and instantly. Computers are not capable of this kind of high-level generalization, so we have to teach them how to do each step in this process separately.

We need to build a pipeline where we solve each step of face recognition separately and pass the result of the current step to the next step. In other words, we will chain together several machine learning algorithms:

Face recognition is really a series of several related problems:

•	First, look at a picture and find all the faces in it
•	Second, focus on each face and be able to understand that even if a face is turned in a weird direction or in bad lighting, it is still the same person.
•	Third, be able to pick out unique features of the face that you can use to tell it apart from other people
•	Finally, compare the unique features of that face to all the people you already know to determine the person’s name

Step 1: Face Detection using HOG (Histogram of Oriented Gradient)

•	To find faces in an image, start by making our image black and white

•	Now for every single pixel, look at the pixels that directly surround it to figure out how dark the current pixel is compared to the pixels directly surrounding it

•	Next, draw an arrow showing in which direction the image is getting darker

•	Repeat this process for every single pixel in the image, you end up with every pixel being replaced by an arrow.

•	These arrows are called gradients and they show the flow from light to dark across the entire image

•	To find faces in this HOG image, we find the part of image that looks the most similar to a known HOG pattern that was extracted from a bunch of other training faces

Step 2: Face Alignment

•	Faces turned in different directions look totally different to a computer

•	To account for this, warp each picture so that the eyes and lips are always in the sample plane in the image using an algorithm called face landmark estimation

•	The basic idea is to come up with n specific points (called landmarks) that exist on every face

o	 the top of the chin
o	the outside edge of each eye
o	he inner edge of each eyebrow, etc.

•	Now train a machine learning algorithm to be able to find these n specific points on any face:

•	Now that the position of eyes and mouth are known, use basic image transformations like rotation and scale that preserve parallel lines called affine transformations

Step 3: Face Encoding

•	The measurements that seem obvious to us humans don’t really make sense to a computer looking at individual pixels in an image. 

•	The most accurate approach is to let the computer figure out the measurements to collect itself. 

•	Deep learning does a better job than humans at figuring out which parts of a face are important to measure.

•	We are going to train it to generate n measurements for each face.

•	The training process works by looking at 3 face images at a time:
o	Load a training face image of a known person
o	Load another picture of the same known person
o	Load a picture of a totally different person

•	Reducing complicated raw data like a picture into a list of computer-generated numbers (128 in this case) is called Embedding.

Step 4: Finding the person’s name from the encoding

•	Next, find the person in our database of known people who has the closest measurements to our test image.

•	This can be done by using any basic machine learning classification algorithm (SVM used for implementation).

•	Train a classifier that can take in the measurements from a new test image and tells which known person is the closest match.
