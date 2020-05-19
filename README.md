# DAugmentor-SearchSpace
This repository includes DAugmentor search space related architectures. If you are planing to submit your GAN architecture please use the following template.

```python
# Developer Name: John Doe
# Developer GitHub Habdle: @JohnGAN
# Citation of Publication (If available)

#Benchmark Results if available

# Define GAN Class
class SampleGAN:

    def construct_discriminator(self, in_shape=(28, 28, 1)):
        '''
        This function defines the discriminator network of the DC-GAN.
        :return:  d_network
        Model Description
        Model Type: Sequential
        Layers:
        '''

        return d_network

    def construct_generator(self, latent_dim):
        '''
        This function defines the generator network of the DC-GAN.
        :return: g_network
        Model Description
        Model Type: Sequential
        Layers: 
        '''

        return g_network

    def construct_gan(self, g_network, d_network):
        """
        construct_gan function assembles a network using constructed g and d networks
        :param g_network:
        :param d_network:
        :return: gan
        """

        return gan

    def prepare_real_samples(self):
        """
        prepare_real_samples function load the data provider and set
        training and testing dataset
        :return: real_samples
        """
        
        return real_samples

    def generate_real_samples(self, dataset, n_samples):
        """
        select real_data samples
        :param dataset:
        :param n_samples:
        :return: X,Y
        """
        
        return X, y

    def generate_latent_points(self, latent_dim, n_samples):
        """
        generate points in latent space as input for the generator
        :param latent_dim:
        :param n_samples:
        :return: x_input
        """
        
        return x_input

    def generate_fake_samples(self, generator, latent_dim, n_samples):
        """
        use the generator to generate n fake examples, with class labels
        :param generator:
        :param latent_dim:
        :param n_samples:
        :return: X,Y
        """
       
        return X, Y

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
        # priint g, d loss
        # print model summary
        # save the generator model
        

    def generate_latent_points(latent_dim, n_samples):
        """
        generate points in latent space as input for the generator
        :param n_samples:
        :return: x_input
        """
        
        return x_input

    def show_plot(self, examples, n):
        """
        create and save a plot of generated images in reversed grayscale
        :param examples:
        :param n:
        :return: none
        """
        # plot images
        

    def generate(self):
        """
        To generate data using the generator model
        """

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

```


