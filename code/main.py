
""" 
Main script
"""

import trainer
import data_utils
import yaml
import argparse
import torch


def parse_config(config_file):
	config = yaml.safe_load(open(config_file, 'r')) 
	return config


if __name__ == '__main__':

	# Arguments
	ap = argparse.ArgumentParser()
	ap.add_argument('-c', '--config', required=True, help='Path to configuration file')
	args = vars(ap.parse_args())

	# Load config file
	config = parse_config(args['config'])

	# Load the dataloaders
	train_loader, val_loader = data_utils.get_dataloader({**config['dataset'], 'batch_size': config['batch_size']})

	# Define the trainer
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"\n[INFO] Found device: {device}")
	model = trainer.Trainer(config, device)

	# Train!
	model.train(train_loader, val_loader)
