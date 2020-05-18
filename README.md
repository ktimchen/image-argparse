# image-argparse
This is an image classifier for this dataset: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. It runs from a command line using argparse and pytorch. 

The first file, train.py, trains a classifier (vgg or aaa) and saves the trained model as a checkpoint. The second file, predict.py, uses the saved model to predict the class for an input image. 
