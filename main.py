import os
import argparse
import yaml
import shutil
import os.path as osp


def make_train_directory(config, path):
    # Create directories if not exist.
    if not os.path.exists(config['TRAINING_CONFIG']['TRAIN_DIR']):
        os.makedirs(config['TRAINING_CONFIG']['TRAIN_DIR'])
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['LOG_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['LOG_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['SAMPLE_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['SAMPLE_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['RESULT_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['RESULT_DIR']))
    if not os.path.exists(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['MODEL_DIR'])):
        os.makedirs(os.path.join(config['TRAINING_CONFIG']['TRAIN_DIR'], config['TRAINING_CONFIG']['MODEL_DIR']))

    shutil.copy(path, osp.join(config['TRAINING_CONFIG']['TRAIN_DIR'], osp.basename(path)))


def main(config, path):

    assert config['TRAINING_CONFIG']['MODE'] in ['train', 'test', 'comb']

    print('{} is started'.format(config['TRAINING_CONFIG']['MODE']))
    if path.startswith('refine'):
        from refine import Refiner
        refiner = Refiner(config)
        if config['TRAINING_CONFIG']['MODE'] == 'train':
            refiner.train()
    elif path.startswith('vis'):
        from vis_image import Vis
        vis = Vis(config)
        vis.test()
    elif path.startswith('tsne'):
        from tsne_vis import TSNEVIS
        vis_tsne = TSNEVIS(config)
        vis_tsne.test()
    elif path.startswith('comb'):
        from comb_analysis import COMB
        comber = COMB(config)
        comber.run()
    elif path.startswith('erra'):
        from erra import ERRORA
        error_a = ERRORA(config)
        error_a.run()
    else:
        from solver import Solver
        solver = Solver(config)
        if config['TRAINING_CONFIG']['MODE'] == 'train':
            solver.train()
    print('{} is finished'.format(config['TRAINING_CONFIG']['MODE']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='exp_config/config_0.yml', help='specifies config yaml file')

    params = parser.parse_args()

    assert osp.exists(params.config)
    config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
    make_train_directory(config, params.config)
    main(config, osp.basename(params.config))


