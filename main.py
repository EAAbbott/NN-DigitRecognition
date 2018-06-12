
import numpy as np
import matplotlib.pyplot as plt
from time import time
from MNIST_extract import mnist_read
from network import N_network
from scipy import ndimage


# =============================================================================
#       1. Load Data
# =============================================================================
start = time()
# reshaping to 784 dimension array for feed forward process.
tr_data = [i.reshape((784,1))
            for i in (mnist_read("data/train-images-idx3-ubyte") / 255.)]
# reshape label to 10 x 1 array for use in cost function.
tr_label_num = mnist_read("data/train-labels-idx1-ubyte")
tr_label = []
for i in tr_label_num:
    j = np.zeros((10, 1))
    j[i] = 1
    tr_label.append(j)
testing_data = [i.reshape((784,1))
            for i in (mnist_read("data/t10k-images-idx3-ubyte") / 255.)]
testing_label = mnist_read("data/t10k-labels-idx1-ubyte")
img_and_labels = list(zip(tr_data, tr_label))
test_and_label = list(zip(testing_data, testing_label))
end = time()
print("Import: {0:.4f}s\n".format((end - start)))


# =============================================================================
#       2. Train and test N_network
# =============================================================================

# Create network with 784 inputs and 10 outputs (28 x 28 images labelled 0 - 9)
network = N_network([784, 50, 10])

# Initialise network weights/biases. (only need to be done before training)
network.init_weights_rand()

# Train network on import data above. test with test data above.
network.train_network(training_data=img_and_labels,
                      batch_size=10,
                      cycles=20,
                      eta=3.0,
                      test_data=test_and_label)



# =============================================================================
#       3. Network results
# =============================================================================


# import drawn images
images = []
labels = []
for i in range(0, 10):
    images.append(((255 - ndimage.imread("usr_img/{0}.png".format(i),
                                 flatten=True)) / 255).reshape((784, 1)))
    label1 = np.zeros((1, 10))
    label1[0][i] = 1
    labels.append(label1)
    img_and_label1 = list(zip(images, labels))



def show_img(data):
    for index, (image, label) in enumerate(data, 1):
        network_output = network.feed_forward(image)
        plt.subplot(3, 3, index)
        plt.imshow(image.reshape((28, 28)),
                   cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.title("Label: {0}\n Network output: {1}".format(np.argmax(label), 
                                           np.argmax(network_output)))
        plt.axis('off')
        plt.show()

show_img(img_and_labels[:9])
#show_img(img_and_label1)


#np.set_printoptions(precision=2, suppress=True, linewidth=100,
#                    threshold=500)
