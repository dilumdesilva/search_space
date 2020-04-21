# DAugmentor Search Space
# Developer Dilum De Silva
# GAN Type - Modified Deep Convolutional Generative Adversarial Network

# Original Architecture details can be found here

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
import numpy as np
from PIL import Image
import argparse
import math
from utils.data_loader import DataLoader


def construct_discriminator():
    '''
    construct_discriminator function defines the discriminator network of the DC-GAN.
    :return:  d_network

    Model Description
    Model Type: Sequential
    Layers:
    '''

    d_network = Sequential()
    d_network.add(Conv2D(64, (5, 5), input_shape=(28, 28, 1), padding='same'))
    d_network.add(Activation('tanh'))
    d_network.add(MaxPooling2D(pool_size=(2, 2)))
    d_network.add(Conv2D(128, (5, 5)))
    d_network.add(Activation('tanh'))
    d_network.add(MaxPooling2D(pool_size=(2, 2)))
    d_network.add(Flatten())
    d_network.add(Dense(1024))
    d_network.add(Activation('tanh'))
    d_network.add(Dense(1))
    d_network.add(Activation('sigmoid'))
    return d_network


# TODO: send latent dim and set the parm to input dim
def construct_generator(latent_dim):
    '''
    construct_generator function defines the generator network of the DC-GAN.
    :return: g_network
    '''
    g_network = Sequential()
    # 7x7 image foundation
    n_nodes = 128 * 7 * 7
    g_network.add(Dense(input_dim=100, output_dim=1024))
    g_network.add(Activation('tanh'))
    g_network.add(Dense(n_nodes))
    g_network.add(BatchNormalization())
    g_network.add(Activation('tanh'))
    # 14x14 up-sampling
    g_network.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    g_network.add(UpSampling2D(size=(2, 2)))
    g_network.add(Conv2D(64, (5, 5), padding='same'))
    g_network.add(Activation('tanh'))
    g_network.add(UpSampling2D(size=(2, 2)))
    g_network.add(Conv2D(1, (5, 5), padding='same'))
    g_network.add(Activation('tanh'))
    return g_network

def construct_gan(g_network, d_network):
    """
    construct_gan function assembles a network using constructed g and d networks
    :param g_network:
    :param d_network:
    :return: gan
    """
    # defining gan as a sequential model
    gan = Sequential()
    # add g_network and d_network of the gan
    gan.add(g_network)
    # set weights of the d_network not trainable
    d_network.trainable = False
    gan.add(d_network)
    return gan

def prepare_real_samples():
    """
    prepare_real_samples function load the data provider and set
    training and testing dataset

    :return: X_train
    """
    # loading real data
    (X_train, y_train), (X_test, y_test) = DataLoader.load_data()
    # convert from int to float and [0,255] to [-1,1] scaling
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    return X_train


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image





def train_gan(g_network, d_network, assembled_gan, dataset, latent_dim, epochs, BATCH_SIZE):
    """
    train_gan function handles the training process of the assembled GAN
    At the end this function saves
        - trained weights of d and g networks
        - trained d and g network models

    :param g_network:
    :param d_network:
    :param assembled_gan:
    :param dataset:
    :param latent_dim:
    :param epochs:
    :param BATCH_SIZE:
    :return: none
    """

    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_network.compile(loss='binary_crossentropy', optimizer="SGD")
    assembled_gan.compile(loss='binary_crossentropy', optimizer=g_optim)
    d_network.trainable = True
    d_network.compile(loss='binary_crossentropy', optimizer=d_optim)

    print("Training")
    batch_per_epo = int(dataset.shape[0] / BATCH_SIZE)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        print("Number of batches", batch_per_epo)
        for index in range(batch_per_epo):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = dataset[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = g_network.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch) + "_" + str(index) + ".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d_network.train_on_batch(X, y)
            print("batch %d_network d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d_network.trainable = False
            g_loss = assembled_gan.train_on_batch(noise, [1] * BATCH_SIZE)
            d_network.trainable = True
            print("batch %d_network g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g_network.save_weights('generator', True)
                d_network.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = construct_generator()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = construct_discriminator()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        # images gen
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def dcgan_executor():
    """
    executor function workflow

    - set latent space size
    - construct generator and discriminator
    - assemble GAN
    - prepare real data sample
    - set number of epochs and batch size
    - train GAN
    - augment data

    :return: none
    """
    latent_dim = 100
    d_network = construct_discriminator()
    g_network = construct_generator(latent_dim)
    assembled_gan = construct_gan(g_network, d_network)
    dataset = prepare_real_samples()
    epochs = 100
    BATCH_SIZE = 128

    train_gan(g_network, d_network, assembled_gan, dataset, latent_dim, epochs, BATCH_SIZE)
    generate(BATCH_SIZE)



# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str)
#     parser.add_argument("--batch_size", type=int, default=128)
#     parser.add_argument("--nice", dest="nice", action="store_true")
#     parser.set_defaults(nice=False)
#     args = parser.parse_args()
#     return args

# if __name__ == "__main__":
#     args = get_args()
#     if args.mode == "train":
#         train_gan(BATCH_SIZE=args.batch_size)
#     elif args.mode == "generate":
#         generate(BATCH_SIZE=args.batch_size, nice=args.nice)
