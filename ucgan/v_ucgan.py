from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from utils.data_loader import DataLoader
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot


class SPucgan:

    def construct_discriminator(self, in_shape=(28, 28, 1)):
        '''
        This function defines the discriminator network of the DC-GAN.

        :return:  d_network


        Model Description
        Model Type: Sequential
        Layers:
        '''

        d_network = Sequential()
        # downsample
        d_network.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
        d_network.add(LeakyReLU(alpha=0.2))
        # downsample
        d_network.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        d_network.add(LeakyReLU(alpha=0.2))
        # classifier
        d_network.add(Flatten())
        d_network.add(Dropout(0.4))
        d_network.add(Dense(1, activation='sigmoid'))
        # compile d_network
        opt = Adam(lr=0.0002, beta_1=0.5)
        d_network.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return d_network

    def construct_generator(self, latent_dim):
        '''
        This function defines the generator network of the DC-GAN.

        :return:
        '''
        g_network = Sequential()
        # 7x7 image foundation
        n_nodes = 128 * 7 * 7
        g_network.add(Dense(n_nodes, input_dim=latent_dim))
        g_network.add(LeakyReLU(alpha=0.2))
        g_network.add(Reshape((7, 7, 128)))
        # 14x14 up-sampling
        g_network.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        g_network.add(LeakyReLU(alpha=0.2))
        # 28x28 up-sampling
        g_network.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        g_network.add(LeakyReLU(alpha=0.2))
        # generate
        g_network.add(Conv2D(1, (7, 7), activation='tanh', padding='same'))
        return g_network

    # def define_generator(latent_dim):
    #     model = Sequential()
    #     # foundation for 7x7 image
    #     n_nodes = 128 * 8 * 8
    #     model.add(Dense(n_nodes, input_dim=latent_dim))
    #     model.add(LeakyReLU(alpha=0.2))
    #     model.add(Reshape((8, 8, 128)))
    #     # upsample to 16x16
    #     model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    #     model.add(LeakyReLU(alpha=0.2))
    #     # upsample to 32x32
    #     model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    #     model.add(LeakyReLU(alpha=0.2))
    #     # upsample to 64x64
    #     model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    #     model.add(LeakyReLU(alpha=0.2))
    #     # upsample to 128x128
    #     model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    #     model.add(LeakyReLU(alpha=0.2))
    #     # generate
    #     model.add(Conv2D(1, (8, 8), activation='tanh', padding='same'))
    #     return model

    def construct_gan(self, g_network, d_network):
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
        gan.add(d_network)
        # set weights of the d_network not trainable
        d_network.trainable = False
        # setting the optimizer and compile gan
        opt = Adam(lr=0.0002, beta_1=0.5)
        gan.compile(loss='binary_crossentropy', optimizer=opt)
        return gan

    def prepare_real_samples(self):
        """
        prepare_real_samples function load the data provider and set
        training and testing dataset

        :return: X
        """
        # loading real data
        (x_train, _), (_, _) = DataLoader.load_data()
        # adding channels to expand to 3d
        X = expand_dims(x_train, axis=-1)
        # convert from int to float and [0,255] to [-1,1] scaling
        X = X.astype('float32')
        X = (X - 127.5) / 127.5
        return X

    def generate_real_samples(self, dataset, n_samples):
        """
        select real_data samples

        :param dataset:
        :param n_samples:
        :return: X,Y
        """
        # choose random instances
        ix = randint(0, dataset.shape[0], n_samples)
        # select images
        X = dataset[ix]
        # generate class labels
        y = ones((n_samples, 1))
        return X, y

    def generate_latent_points(self, latent_dim, n_samples):
        """
        generate points in latent space as input for the generator
        :param latent_dim:
        :param n_samples:
        :return: x_input
        """
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    def generate_fake_samples(self, generator, latent_dim, n_samples):
        """
        use the generator to generate n fake examples, with class labels
        :param generator:
        :param latent_dim:
        :param n_samples:
        :return: X,Y
        """
        # generate points in latent space
        x_input = SPucgan.generate_latent_points(latent_dim, n_samples)
        # predict outputs
        X = generator.predict(x_input)
        # create class labels
        y = zeros((n_samples, 1))
        return X, y

    def train_gan(self, g_network, d_network, assembled_gan, dataset, latent_dim, epochs=100, BATCH_SIZE=128):
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
        print("Training")
        batch_per_epo = int(dataset.shape[0] / BATCH_SIZE)
        half_batch = int(BATCH_SIZE / 2)

        # enumerate epochs
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            print("Number of batches", batch_per_epo)
            # enumerate batches over the training set
            for index in range(batch_per_epo):
                # get randomly selected 'real_data' samples
                X_real, y_real = SPucgan.generate_real_samples(dataset, half_batch)
                # update discriminator model weights
                d_loss1, _ = d_network.train_on_batch(X_real, y_real)
                # generate 'fake' examples
                X_fake, y_fake = SPucgan.generate_fake_samples(g_network, latent_dim, half_batch)
                # update discriminator model weights
                d_loss2, _ = d_network.train_on_batch(X_fake, y_fake)
                # prepare points in latent space as input for the generator
                X_gan = SPucgan.generate_latent_points(latent_dim, BATCH_SIZE)
                # create inverted labels for the fake samples
                y_gan = ones((BATCH_SIZE, 1))
                # update the generator via the discriminator's error
                g_loss = assembled_gan.train_on_batch(X_gan, y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                      (epoch + 1, index + 1, batch_per_epo, d_loss1, d_loss2, g_loss))
        # save the generator model
        g_network.save('generator.h5')

    def generate_latent_points(latent_dim, n_samples):
        """
        generate points in latent space as input for the generator
        :param n_samples:
        :return: x_input
        """
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    def show_plot(self, examples, n):
        """
        create and save a plot of generated images in reversed grayscale
        :param examples:
        :param n:
        :return: none
        """
        # plot images
        for i in range(n * n):
            # define subplot
            pyplot.subplot(n, n, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
        pyplot.show()

    def generate(self):
        # load model
        model = load_model('generator.h5')
        # generate images
        latent_points = SPucgan.generate_latent_points(100, 100)
        # generate images
        X = model.predict(latent_points)
        # plot the result
        SPucgan.show_plot(X, 10)

    def ucgan_executor(self):
        """
        executor function workflow

        - set latent space size
        - construct generator and discriminator
        - assemble GAN
        - prepare real data sample
        - set number of epochs and batch size
        - train GAN
        - augment data

        :return:none
        """
        # set latent space size
        # construct generator and discriminator
        # assemble GAN
        # prepare real data sample
        # set number of epochs and batch size
        # train GAN
        # augment data

        latent_dim = 100
        d_network = SPucgan.construct_discriminator()
        g_network = SPucgan.construct_generator(latent_dim)
        assembled_gan = SPucgan.construct_gan(g_network, d_network)
        dataset = SPucgan.prepare_real_samples()
        epochs = 100
        BATCH_SIZE = 128
        SPucgan.train_gan(g_network, d_network, assembled_gan, dataset, latent_dim)
        SPucgan.generate()
