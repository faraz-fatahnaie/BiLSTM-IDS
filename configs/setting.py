from pathlib import Path
import json


def setting(config_file=None):
    config = dict()
    BASE_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR.joinpath('session').mkdir(exist_ok=True)

    if config_file is None:
        config_name = 'CONFIG'  # config file name in config dir
        config_dir = BASE_DIR.joinpath('configs')
        config_file = open(f'{config_dir}/{config_name}.json')
        config_file = json.load(config_file)

    config['SEED'] = config_file['seed']

    config['K_FOLD'] = int(config_file['dataset']['k_fold'])
    config['DATASET_TYPE'] = config_file['dataset']['type']
    config['CLASSIFICATION_MODE'] = config_file['dataset']['classification_mode']

    config['NUM_WORKER'] = config_file['dataset']['n_worker']
    config['MODEL_NAME'] = config_file['model']['name']

    config['EPOCHS'] = config_file['model']['epoch']
    config['BATCH_SIZE'] = config_file['model']['batch_size']
    config['LOSS'] = config_file['model']['loss']
    config['OPTIMIZER'] = config_file['model']['optimizer']
    config['LR'] = config_file['model']['lr']
    config['MIN_LR'] = config_file['model']['min_lr']
    config['PATIENCE'] = config_file['model']['patience']


    return config, config_file
