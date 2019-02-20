from collections import OrderedDict

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from fastai.vision import create_head

# source: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
        
    def forward(self, x):
        return self.f(x)
        
class DynamicDuo(nn.Module):
    """Creates a duo network composed of `hero` alongisde a `sidekick_scale` copy of `hero`."""
    def __init__(self, hero, num_classes=1000, sidekick_scale=0.25):
        super(DynamicDuo, self).__init__()
        
        self.hero = hero()
        # load pretrained weights
        self.hero.load_pretrained_weights()
        # freeze hero
        for params in self.hero.parameters():
            params.requires_grad = False
        self.sidekick = hero(num_classes=num_classes, sidekick_scale=sidekick_scale, mode='sidekick')
    
    def forward(self, x):
        x0 = x
        
        hero_activations = {
            'body': [],
            'head': [],
        }
        
        for hero_block in self.hero.body:
            x = hero_block(x)
            hero_activations['body'].append(x)
        
        # ALT: Run sidekick on fc layers too
#         for hero_block in self.hero.head:
#             x = hero_block(x)
#             hero_activations['head'].append(x)
            
        x = self.sidekick.body[0](x0)
        for sidekick_block, hero_activation in zip(self.sidekick.body[1:], hero_activations['body']):
            x = sidekick_block(torch.cat([hero_activation, x], dim=1))
        x = torch.cat([hero_activations['body'][-1], x], dim=1)
        
        # ALT: Run sidekick on fc layers too
#         x = self.sidekick.head[0](x)
#         for sidekick_block, hero_activation in zip(self.sidekick.head[1:], hero_activations['head']):
#             x = sidekick_block(torch.cat([hero_activation, x], dim=1))
#         return x

        # DEFAULT: Run sidekick through fastai head
        x = self.sidekick.head(x)
        
        return x
        
class AlexNet(nn.Module):
    """An AlexNet suitable for 'default', 'finetune' and 'sidekick' modes."""
    def __init__(self, num_classes=1000, sidekick_scale=0.25, mode='default'):
        super(AlexNet, self).__init__()
                 
        num_channels = [3, 64, 192, 384, 256, 256]
        num_features = [256, 4096, 4096]
        
        nc_in = num_channels[:-1]
        nc_out = num_channels[1:]
        nf_in = num_features
        nf_out = num_features[1:]
        
        if mode == 'sidekick':
            nc_in[1:] = [int((1.0 + sidekick_scale) * nc) for nc in num_channels[1:-1]]
            nc_out = [int(sidekick_scale * nc) for nc in nc_out]    
            nf_in = [int((1.0 + sidekick_scale) * nf) for nf in nf_in]
            nf_out = [int(sidekick_scale * nf) for nf in nf_out]
        
        self.body = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(nc_in[0], nc_out[0], kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(nc_in[1], nc_out[1], kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
            nn.Sequential(
                nn.Conv2d(nc_in[2], nc_out[2], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(nc_in[3], nc_out[3], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(nc_in[4], nc_out[4], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ),
        )
        
        self.head = nn.Sequential(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((6, 6)),
                Flatten(),
                nn.Dropout(),
                nn.Linear(nf_in[0] * 6 * 6, nf_out[0]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(nf_in[1], nf_out[1]),
                nn.ReLU(inplace=True),
            ),
            nn.Linear(nf_in[2], num_classes),
        )
        
        # Comment out this replacement head if you want to use ALT path in DynamicDuo
        if mode in ['sidekick', 'finetune']:
            self.head = create_head(nf_in[0]*2, num_classes)

        self.mode = mode
        
    def load_pretrained_weights(self):    
        if self.mode == 'sidekick':
            raise Exception('This network appears to be a sidekick. It cannot be pretrained.')
        pretrained_state_dict = model_zoo.load_url(model_urls['alexnet'])
        state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        
        keys = list(state_dict.keys())
        if self.mode == 'finetune':
            # ignore the pretrained head
            keys = [key for key in keys if 'body' in key]
        for key, pretrained_key in zip(keys, pretrained_state_dict.keys()):
            new_state_dict[key] = pretrained_state_dict[pretrained_key]
        state_dict.update(new_state_dict)
        self.load_state_dict(state_dict)
   
    def prepare_to_finetune(self):
        if not self.mode == 'finetune':
            raise Exception('This network is not in "finetune" mode.')
        self.load_pretrained_weights()
        for params in self.body.parameters():
            params.requires_grad = False

    def forward(self, x):
        if self.mode == 'sidekick':
            raise Exception('This network appears to be a sidekick. It can only operate as part of a duo.')
        x = self.body(x)
        x = self.head(x)
        return x
