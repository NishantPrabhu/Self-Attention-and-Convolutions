
""" 
Main script
"""

import trainer
import data_utils
import yaml
import argparse
import torch
import os
from datetime import datetime as dt


def parse_config(config_file):
	config = yaml.safe_load(open(config_file, 'r')) 
	return config

def create_output_dir(name):
	if not os.path.exists(name):
		os.makedirs(name, exist_ok=True)
	return name


if __name__ == '__main__':

	# Arguments
	ap = argparse.ArgumentParser()
	ap.add_argument('-c', '--config', required=True, help='Path to configuration file')
	ap.add_argument('-o', '--output', default='../'+dt.now().strftime("%Y-%m-%d_%H-%M"), help='Output directory')
	args = vars(ap.parse_args())

	# Load config file
	config = parse_config(args['config'])
	output_dir = create_output_dir(args['output'])

	# Load the dataloaders
	train_loader, val_loader = data_utils.get_dataloader({**config['dataset'], 'batch_size': config['batch_size']})

	# Define the trainer
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"\n[INFO] Found device: {torch.cuda.get_device_name(0)}")
	model = trainer.Trainer(config, output_dir, device)

	# Train!
	model.train(train_loader, val_loader)
