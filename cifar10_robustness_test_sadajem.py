import torch
import torch.nn as nn
import argparse
from autoattack import AutoAttack
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
from models.jem_models import CCF

class NormalizationWrapper(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()

        mean = mean[..., None, None]
        std = std[..., None, None]

        self.train(model.training)

        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x, *args, **kwargs):
        x_normalized = (x - self.mean)/self.std
        return self.model(x_normalized, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict()

def get_CIFAR10(batch_size=128, num_workers=2):
    """
    Simplified CIFAR-10 test dataset loader for robustness testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10('./data', train=False, 
                              transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)
    return loader


parser = argparse.ArgumentParser(description='SADAJEM CIFAR-10 Robustness Evaluation', prefix_chars='-')

parser.add_argument('--gpu','--list', nargs='+', default=list(range(torch.cuda.device_count())),
                    help='GPU indices, if more than 1 parallel modules will be called')
parser.add_argument('--distance_type', type=str, default='L2', choices=['L2', 'Linf'],
                    help='Distance type for adversarial attacks (L2 or Linf)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='Epsilon value for adversarial attacks (default: 0.5)')

hps = parser.parse_args()

if len(hps.gpu)==0:
    device = torch.device('cpu')
    print('Warning! Computing on CPU')
    num_devices = 1
elif len(hps.gpu)==1:
    device_ids = [int(hps.gpu[0])]
    device = torch.device('cuda:' + str(hps.gpu[0]))
    num_devices = 1
else:
    device_ids = [int(i) for i in hps.gpu]
    device = torch.device('cuda:' + str(min(device_ids)))
    num_devices = len(device_ids)

ROBUSTNESS_DATAPOINTS = 100_000
dataset = 'cifar10'

bs = 100 * num_devices

print(f'Testing on {ROBUSTNESS_DATAPOINTS} points from {dataset.upper()} dataset')

# Load SADAJEM model
model = CCF(depth=28, width=10, norm='batch', n_classes=10, model='wrn')
ckpt_dict = torch.load("./sadajem5_95.5_9.4.pt")
if isinstance(ckpt_dict, dict) and "model_state_dict" in ckpt_dict:
    model.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]
else:
    model.load_state_dict(ckpt_dict)
    replay_buffer = None
mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])
model = NormalizationWrapper(model, mean, std)
model.to(device)

if len(hps.gpu) > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

model.eval()
print('SADAJEM model loaded successfully')

# Load data for robustness evaluation
dataloader = get_CIFAR10(batch_size=ROBUSTNESS_DATAPOINTS, num_workers=2)
data_iterator = iter(dataloader)
ref_data, target = next(data_iterator)

print(f'Distance type: {hps.distance_type}, Eps: {hps.eps}')

# Run AutoAttack evaluation
num_classes = 10  # CIFAR-10 has 10 classes

if hps.distance_type == 'L2':
    attack = AutoAttack(model, device=device, norm='L2', eps=hps.eps, verbose=True)
    # Set the number of target classes for targeted attacks
    attack.apgd_targeted.n_target_classes = num_classes - 1
    attack.fab.n_target_classes = num_classes - 1
    attack.run_standard_evaluation(ref_data, target, bs=bs)
elif hps.distance_type == 'Linf':
    attack = AutoAttack(model, device=device, norm='Linf', eps=hps.eps, verbose=True)
    # Set the number of target classes for targeted attacks
    attack.apgd_targeted.n_target_classes = num_classes - 1
    attack.fab.n_target_classes = num_classes - 1
    attack.run_standard_evaluation(ref_data, target, bs=bs)

