"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from utils.config import *
from datasets import VRC_data
from agents import *
from graphs.models import VRC_model
import torch
from utils import general_loss as gnl
def datatest(config):
    data_loader = VRC_data.VRC_DataLoader(config).dataset
    print(data_loader[0])
def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()
    args.config = 'configs/vrc_exp_0.json'
    # parse the config json file
    config = process_config(args.config)
    decoder = VRC_model.Decoder(config)
    pretrained_dict = torch.load(config.LocalGrid_pretrian_file)
    model_dict = decoder.state_dict()
    update_dict = {k[8:] : v for k, v in pretrained_dict.items()}
    model_dict.update(update_dict)
    decoder.load_state_dict(model_dict)
    print(gnl.prior_regular(decoder,decoder))
main()