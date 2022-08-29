"""
This module is used for GAN creation and training.
"""



# Imports
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import VanilllaGAN, WGAN, WGAN_GP, CRAMERGAN



class ModelTrainer:
    """
    A class used for the creation and training of GAN models.


    Attributes:
    -----------
    real_data: pandas.core.frame.DataFrame
        A dataframe containing the data used to train the GANs.
    batch_size: int
        An integer defining the size of the batch.
    num_cols: list
        A list containing all integer columns.
    cat_cols: list
        A list containing all categorical columns.
    noise_dim: int
        An integer defining the number of input nodes to the Generator.
    dim: int
        An integer containing the layer information.
    data_dim: int
        An integer defining the size of the data (number of columns).
    log_step: int
        An integer defining the size of the log steps.
    epochs: int
        An integer defining the number of epochs.
    learning_rate: float
        An integer defining the learning rate of the GANs.
    beta_1: float
        Defines beta1 of the Adam optimizer.
    beta_2: float
        Defines beta2 of the Adam optimizer.


    Methods:
    --------
    gan_training()
        Create and train a GAN algorithm.
    wgan_training()
        Create and train a WGAN algorithm.
    wgangp_training()
        Create and train a WGAN-GP algorithm.
    cramergan_training()
        Create and train a CramerGAN algorithm.
    """



    def __init__(self, real_data, batch_size):
        """
        Initialization of the class.


        Parameters
        ----------
        real_data: pandas.core.frame.DataFrame
            The real data used to train the GANs.
        """



        # Setting the initial parameters
        self.real_data = real_data
        self.num_cols = real_data.select_dtypes(include=['float64']).columns.tolist()
        self.cat_cols = real_data.select_dtypes(include=['category']).columns.tolist()
        self.noise_dim = 5#32
        self.dim = 30#128
        self.data_dim = real_data.shape[1]
        self.batch_size = batch_size # 128 or 16 in paper --> Up to change
        self.log_step = 100
        self.epochs = 200 # Up to change
        self.learning_rate = 5e-4
        self.beta_1 = 0.5
        self.beta_2 = 0.9



    def gan_training(self):
        """
        Train a GAN.


        Returns
        -------
        pandas.core.frame.DataFrame
            Returns the Generator of the GAN algorithm.
        """


        # Define the parameters used to set up the GAN model
        gan_args = ModelParameters(batch_size=self.batch_size,
                                   lr=self.learning_rate,
                                   betas=(self.beta_1, self.beta_2),
                                   noise_dim=self.noise_dim,
                                   layers_dim=self.dim)

        # Define the parameters used for GAN training
        train_args = TrainParameters(epochs = self.epochs,
                                     sample_interval = self.log_step)

        # Select the GAN model
        model = VanilllaGAN

        # Initialize the GAN model with the pre-defined parameters
        synthesizer = model(gan_args)

        # Train the GAN model
        synthesizer.train(self.real_data, train_args,
                            (self.real_data.select_dtypes(include=['float64'])).columns.tolist(),
                            (self.real_data.select_dtypes(include=['category'])).columns.tolist())

        # Return the Generator of the GAN model
        return synthesizer



    def wgan_training(self):
        """
        Train a WGAN.


        Returns
        -------
        pandas.core.frame.DataFrame
            Returns the Generator of the WGAN algorithm.
        """

        # Define the parameters used to set up the WGAN model
        gan_args = ModelParameters(batch_size=self.batch_size,
                                   lr=self.learning_rate,
                                   betas=(self.beta_1, self.beta_2),
                                   noise_dim=self.noise_dim,
                                   layers_dim=self.dim)

        # Define the parameters used for WGAN training
        train_args = TrainParameters(epochs = self.epochs,
                                     sample_interval = self.log_step)

        # Select the WGAN model
        model = WGAN

        # Initialize the WGAN model with the pre-defined parameters
        synthesizer = model(gan_args, n_critic=2)

        # Train the WGAN model
        synthesizer.train(self.real_data, train_args,
                            (self.real_data.select_dtypes(include=['float64'])).columns.tolist(),
                            (self.real_data.select_dtypes(include=['category'])).columns.tolist())

        # Return the Generator of the WGAN model
        return synthesizer



    def wgangp_training(self):
        """
        Train a WGAN-GP.


        Returns
        -------
        pandas.core.frame.DataFrame
            -Returns the generator of the WGAN-GP algorithm.
        """

        # Define the parameters used to set up the WGAN-GP model
        gan_args = ModelParameters(batch_size=self.batch_size,
                                   lr=self.learning_rate,
                                   betas=(self.beta_1, self.beta_2),
                                   noise_dim=self.noise_dim,
                                   layers_dim=self.dim)

        # Define the parameters used for WGAN-GP training
        train_args = TrainParameters(epochs=self.epochs,
                                     sample_interval=self.log_step)

        # Select the WGAN-GP model
        model = WGAN_GP

        # Initialize the WGAN-GP model with the pre-defined parameters
        synthesizer = model(gan_args, n_critic=2)

        # Train the WGAN-GP model
        synthesizer.train(self.real_data, train_args,
                            (self.real_data.select_dtypes(include=['float64'])).columns.tolist(),
                            (self.real_data.select_dtypes(include=['category'])).columns.tolist())

        # Return the Generator of the WGAN-GP model
        return synthesizer



    def cramergan_training(self):
        """
        Train a CramerGAN.


        Returns
        -------
        pandas.core.frame.DataFrame
            Returns the generator of the CramerGAN algorithm.
        """

        # Define the parameters used to set up the CramerGAN model
        gan_args = ModelParameters(batch_size=self.batch_size,
                                   lr=self.learning_rate,
                                   betas=(self.beta_1, self.beta_2),
                                   noise_dim=self.noise_dim,
                                   layers_dim=self.dim)

        # Define the parameters used for CramerGAN training
        train_args = TrainParameters(epochs=self.epochs,
                                     sample_interval=self.log_step)

        # Select the CramerGAN model
        model = CRAMERGAN

        # Initialize the CramerGAN model with the pre-defined parameters
        synthesizer = model(gan_args)

        # Train the CramerGAN model
        synthesizer.train(self.real_data, train_args,
                          (self.real_data.select_dtypes(include=['float64'])).columns.tolist(),
                          (self.real_data.select_dtypes(include=['category'])).columns.tolist())

        # Return the Generator of the CramerGAN model
        return synthesizer
