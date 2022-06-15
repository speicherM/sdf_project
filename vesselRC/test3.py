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
    datatest(config)
main()