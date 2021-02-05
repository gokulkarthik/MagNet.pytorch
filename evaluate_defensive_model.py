import logging
import os
import yaml

import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from classifiers import Classifier1, Classifier2
from defensive_models import DefensiveModel1, DefensiveModel2

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['evaluate_defensive_model']
dataset_name = config['dataset_name']
defensive_models_path = config['defensive_models_path']
defensive_model_name = config['defensive_model_name']
classifier_models_path = config['classifier_models_path']
classifier_model_name = config['classifier_model_name']
attacks_data_path = config['attacks_data_path']
attack_model_name = config['attack_model_name']
batch_size = config['batch_size']
result_path = config['result_path']
visualize = config['visualize']
visualization_path = config['visualization_path']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if not os.path.exists(result_path):
    os.mkdir(result_path)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    Thanks: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def get_defensive_model(defensive_models_path, defensive_model_name, dataset_name):
    logging.info("Entering the function 'get_defensive_model' in 'evaluate_defensive_model.py'")
    global device

    if defensive_model_name == 'defensive-model-1':
        model = DefensiveModel1()
    elif defensive_model_name == 'defensive-model-2':
        model = DefensiveModel2()
    else:
        raise ValueError("Undefined defensive model")
    model_state = torch.load(os.path.join(defensive_models_path,
                                          f'{dataset_name}_{defensive_model_name}_checkpoint.pth'))['state_dict']
    model.load_state_dict(model_state)
    model.eval().to(device)

    logging.info("Exiting the function 'get_defensive_model' in 'evaluate_defensive_model.py'")
    return model


def get_classifier_model(classifier_models_path, classifier_model_name, dataset_name):
    logging.info("Entering the function 'get_classifier_model' in 'evaluate_defensive_model.py'")
    global device

    if classifier_model_name == 'classifier-1':
        model = Classifier1()
    elif classifier_model_name == 'classifier-2':
        model = Classifier2()
    else:
        raise ValueError("Undefined defensive model")
    model_state = torch.load(os.path.join(classifier_models_path,
                                          f'{dataset_name}_{classifier_model_name}_checkpoint.pth'))['state_dict']
    model.load_state_dict(model_state)
    model.eval().to(device)

    logging.info("Exiting the function 'get_classifier_model' in 'evaluate_defensive_model.py'")
    return model


def get_transform(dataset_name):
    if dataset_name in ['mnist', 'fashion-mnist']:
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    elif dataset_name in ['cifar-10']:
        return transforms.Compose([
            transforms.ToTensor()
        ])

    return -1


def compute_reconstruction_error(img_mb_1, img_mb_2, ord_='l2'):
    img_mb_1 = img_mb_1.view(img_mb_1.shape[0], -1)
    img_mb_2 = img_mb_2.view(img_mb_2.shape[0], -1)
    diff_mb = torch.abs(img_mb_1 - img_mb_2)
    if ord_ == 'l0':
        reconstruction_error_mb = (diff_mb != 0).float().mean(dim=1)
    elif ord_ == 'l1':
        reconstruction_error_mb = diff_mb.mean(dim=1)
    elif ord_ == 'l2':
        reconstruction_error_mb = (diff_mb ** 2).mean(dim=1)
    elif ord_ == 'linf':
        reconstruction_error_mb, _ = diff_mb.max(dim=1)
    else:
        reconstruction_error_mb = -1

    return reconstruction_error_mb


defensive_model = get_defensive_model(defensive_models_path, defensive_model_name, dataset_name)
classifier_model = get_classifier_model(classifier_models_path, classifier_model_name, dataset_name)

first_time = True
result_df_path = os.path.join(result_path, f'{dataset_name}_{attack_model_name}_{defensive_model_name}.csv')
attacks_path = os.path.join(attacks_data_path, dataset_name, attack_model_name)
attacks = sorted(os.listdir(attacks_path))

for attack_idx, attack in tqdm(enumerate(attacks), leave=True, desc='Evaluating Attacks:', total=len(attacks)):

    attack_dataset_path = os.path.join(attacks_path, attack)

    with open(os.path.join(attack_dataset_path, 'config.yaml'), 'r') as file:
        attack_config = yaml.load(file, Loader=yaml.Loader)

    dataset = ImageFolderWithPaths(attack_dataset_path, transform=get_transform(dataset_name))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for X_mb, Y_mb, P_mb in tqdm(data_loader, leave=False, desc=f'{attack}'):

        X_mb, Y_mb = X_mb.to(device), Y_mb.to(device)

        X_purified_mb = defensive_model(X_mb)
        reconstruction_error_l0_mb = compute_reconstruction_error(X_mb, X_purified_mb, ord_='l0')
        reconstruction_error_l1_mb = compute_reconstruction_error(X_mb, X_purified_mb, ord_='l1')
        reconstruction_error_l2_mb = compute_reconstruction_error(X_mb, X_purified_mb, ord_='l2')
        reconstruction_error_linf_mb = compute_reconstruction_error(X_mb, X_purified_mb, ord_='linf')
        Y_pred_mb = classifier_model(X_mb).argmax(1)
        Y_pred_purified_mb = classifier_model(X_purified_mb).argmax(1)

        attack_mb_result_df = pd.DataFrame({
            'path': P_mb,
            'Y': Y_mb.flatten().tolist(),
            'Y_pred': Y_pred_mb.flatten().tolist(),
            'Y_pred_purified': Y_pred_purified_mb.flatten().tolist(),
            'reconstruction_error_l0': reconstruction_error_l0_mb.flatten().tolist(),
            'reconstruction_error_l1': reconstruction_error_l1_mb.flatten().tolist(),
            'reconstruction_error_l2': reconstruction_error_l2_mb.flatten().tolist(),
            'reconstruction_error_linf': reconstruction_error_linf_mb.flatten().tolist(),
        })
        attack_mb_result_df['attack'] = attack
        attack_mb_result_df['is_correct_without_defense'] = attack_mb_result_df['Y'] == attack_mb_result_df['Y_pred']
        attack_mb_result_df['is_correct_with_defense'] = attack_mb_result_df['Y'] == attack_mb_result_df['Y_pred_purified']

        if first_time:
            attack_mb_result_df.to_csv(result_df_path, mode='w', header=True, index=False, float_format='%.4f')
            first_time = False
        else:
            attack_mb_result_df.to_csv(result_df_path, mode='a', header=False, index=False, float_format='%.4f')

        if visualize:
            X_diff_mb = X_mb - X_purified_mb
            X_diff_abs_mb = torch.abs(X_diff_mb)
            for i in tqdm(range(len(P_mb)), leave=False, desc="Saving Visualizations:"):
                images = torch.stack([X_mb[i], X_purified_mb[i], X_diff_mb[i], X_diff_abs_mb[i]])
                visualization = torchvision.utils.make_grid(images, nrow=4)
                grid_dir = os.path.join(visualization_path, '/'.join(P_mb[i].split('/')[1:-1]))
                if not os.path.exists(grid_dir):
                    os.makedirs(grid_dir)
                image_fn = P_mb[i].split('/')[-1]
                image_fn_new = image_fn.replace('.', f'_{Y_pred_mb[i]}_{Y_pred_purified_mb[i]}.')
                grid_path = os.path.join(grid_dir, image_fn_new)
                torchvision.utils.save_image(visualization, grid_path)


result_df = pd.read_csv(result_df_path)
result_df = result_df.rename(columns={'is_correct_without_defense': 'accuracy_without_defense',
                                      'is_correct_with_defense': 'accuracy_with_defense'})
cols = ['reconstruction_error_l0', 'reconstruction_error_l1', 'reconstruction_error_l2', 'reconstruction_error_linf',
        'accuracy_without_defense', 'accuracy_with_defense']
result_df_summary = result_df.groupby('attack')[cols].mean()
result_df_summary_path = result_df_path.replace('.', '_summary.')
result_df_summary.to_csv(result_df_summary_path, index=True)