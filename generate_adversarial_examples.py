import logging
import os
import shutil
import yaml

import foolbox as fb
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from classifiers import Classifier1, Classifier2

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['generate_adversarial_examples']
seed = config['seed']
save_path = config['save_path']
dataset = config['dataset']
attack_model = config['attack_model']

batch_size = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

save_path = os.path.join(save_path, dataset, attack_model)
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)


def get_dataset(dataset):
    logging.info("Entering the function 'get_dataset' in 'generate_adversarial_examples.py'")

    if dataset == 'mnist':
        test_set = datasets.MNIST('../data', train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                  ]))

    elif dataset == 'fashion-mnist':
        test_set = datasets.FashionMNIST('../data', train=False, download=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))

    elif dataset == 'cifar-10':
        test_set = datasets.CIFAR10('../data/', train=False, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                    ]))

    else:
        raise ValueError("Undefined dataset")

    logging.info("Exiting the function 'get_dataset' in 'generate_adversarial_examples.py'")
    return test_set


def get_model(attack_model):
    logging.info("Entering the function 'get_model' in 'generate_adversarial_examples.py'")

    if attack_model == 'classifier-1':
        model = Classifier1()
    elif attack_model == 'classifier-2':
        model = Classifier2()
    else:
        raise ValueError("Undefined classifier")
    model_state = torch.load(os.path.join('models', f'{dataset}_{attack_model}_checkpoint.pth'))['state_dict']
    model.load_state_dict(model_state)
    model.eval()

    logging.info("Exiting the function 'get_model' in 'generate_adversarial_examples.py'")
    return model


def make_dirs(config, test_set):
    logging.info("Entering the function 'make_dirs' in 'generate_adversarial_examples.py'")
    global save_path

    attack_save_path = os.path.join(save_path, f"Attack-{config['attack_id']}")
    os.mkdir(attack_save_path)

    with open(os.path.join(attack_save_path, 'config.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    if config['dataset'] in ['mnist', 'fashion-mnist']:
        unique_targets = [unique_target.item() for unique_target in torch.unique(test_set.targets)]
    elif config['dataset'] in ['cifar-10']:
        unique_targets = list(set(test_set.targets))
    for unique_target in unique_targets:
        os.mkdir(os.path.join(attack_save_path, str(unique_target)))

    logging.info("Exiting the function 'make_dirs' in 'generate_adversarial_examples.py'")
    return True


def make_adversarial_examples(config, test_set):
    logging.info("Entering the function 'make_adversarial_examples' in 'generate_adversarial_examples.py'")
    global save_path

    attack_save_path = os.path.join(save_path, f"Attack-{config['attack_id']}")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    for batch_idx, (X_mb, Y_mb) in tqdm(enumerate(test_loader), leave=False, total=len(test_loader),
                                        desc=f"Attack: {config['attack_id']}"):
        X_mb, Y_mb = X_mb.to(device), Y_mb.to(device)
        attack_function = config['attack_function']
        _, adversarials_mb, success_mb = attack_function(fmodel, X_mb, Y_mb, epsilons=config['epsilons'])
        for i in tqdm(range(len(X_mb)), leave=False, desc=f"Attack: {config['attack_id']}; Batch: {batch_idx}"):
            adversarial_image = adversarials_mb[i]
            y = Y_mb[i].item()
            image_id = batch_idx * batch_size + i
            attack_image_save_path = os.path.join(attack_save_path, str(y), f'{image_id}.png')
            save_image(adversarial_image, attack_image_save_path)

    logging.info("Entering the function 'make_adversarial_examples' in 'generate_adversarial_examples.py'")
    return True


test_set = get_dataset(dataset)
model = get_model(attack_model)
fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=dict())


# Attack: 0
# No attack
######################
config['attack_id'] = 0
config['attack_function'] = None
config['epsilons'] = 0

make_dirs(config, test_set)

attack_save_path = os.path.join(save_path, f"Attack-{config['attack_id']}")
for i in tqdm(range(len(test_set)), leave=False, desc=f"Attack: {config['attack_id']}"):
    x, y = test_set[i]
    attack_image_save_path = os.path.join(attack_save_path, str(y), f'{i}.png')
    save_image(x, attack_image_save_path)


# Attack: 1
# FGSM
######################
config['attack_id'] = 1
config['attack_function'] = fb.attacks.FGSM()
config['epsilons'] = 0.01

make_dirs(config, test_set)

make_adversarial_examples(config, test_set)


# Attack: 2
# FGSM
######################
config['attack_id'] = 2
config['attack_function'] = fb.attacks.FGSM()
config['epsilons'] = 0.1

make_dirs(config, test_set)

make_adversarial_examples(config, test_set)


# Attack: 3
# L2-PGD
######################
config['attack_id'] = 3
config['attack_function'] = fb.attacks.L2PGD()
config['epsilons'] = 0.01

make_dirs(config, test_set)

make_adversarial_examples(config, test_set)

# Attack: 4
# L2-PGD
######################
config['attack_id'] = 4
config['attack_function'] = fb.attacks.L2PGD()
config['epsilons'] = 0.1

make_dirs(config, test_set)

make_adversarial_examples(config, test_set)


# Attack: 5
# L2-PGD
######################
config['attack_id'] = 5
config['attack_function'] = fb.attacks.L2PGD()
config['epsilons'] = 0.5

make_dirs(config, test_set)

make_adversarial_examples(config, test_set)


# Attack: 6
# DeepFool
######################
config['attack_id'] = 6
config['attack_function'] = fb.attacks.L2DeepFoolAttack()
config['epsilons'] = 0.1

make_dirs(config, test_set)

make_adversarial_examples(config, test_set)


# Attack: 7
# Carlini-Wagner
######################
config['attack_id'] = 7
config['attack_function'] = fb.attacks.L2CarliniWagnerAttack(steps=1000)
config['epsilons'] = 0.1

make_dirs(config, test_set)

make_adversarial_examples(config, test_set)