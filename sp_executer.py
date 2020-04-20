# execute the GANs in the search space one by one
# Show which GAN is currently processing

from utils import constants as const
from utils.data_loader import DataLoader

class SPExecuter:
    def __init__(self, path):
        self.path = path

    def set_pre_processing_data_path(self):
        print("Data path has been set")

    def execute_sp(self):
        print("UC-GAN Assembled")
        SPExecuter.ucgan_runner()
        print("DC-GAN Assembled")
        SPExecuter.dcgan_runner()

    def ucgan_runner(self):
        # UC-GAN running
        print("SP-Runner: UCGAN")

    def dcgan_runner(self):
        # DC-GAN running
        print("SP-Runner: DCGAN")

    # write gan related time batch size epochs shape to a csv

    def construct_data_for_mc_calculation(self):
        print("MC score identifier started")
        # prepare data for UCGAN
        # Prepare data for DCGAN

        # pass the two arrays to the mc identifier

