# 2022. 1. 10. Ver007 FrEIA version 0.1, 0.2
import torch
import torch.nn as nn
import torch.optim
# from torch.autograd import Variable

# from FrEIA.framework import *
# from FrEIA.modules import *
# import FrEIA0p1.framework as Ff0p1
# import FrEIA0p1.modules as Fm0p1
# import FrEIA0p2.framework as Ff0p2
# import FrEIA0p2.modules as Fm0p2
from ..FrEIA import framework as Ff
from ..FrEIA import modules as Fm


class FTransformNet_GLOW(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        self.model = self.build_network(c)
        # torch.manual_seed(1)
        torch.manual_seed(c.seed_weight_init)
        self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        for p in self.params_trainable:
             p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)
        self.optim = torch.optim.Adam(self.params_trainable, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
        self.weight_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=c.meta_epoch, gamma=c.gamma)
        
    def build_network(self, c):
        # model = fully_connected(c.y_dim_features, c.y_dim_features, width=128, n_layer=3)
        
        INN = Ff.GraphINN
        
        def fc_constr(c_in, c_out):
            layers = [ nn.Linear(c_in, c.internal_width), nn.ReLU() ]
            for i_layer in range(1, c.internal_layer-1):
                layers += [ nn.Linear(c.internal_width,  c.internal_width), nn.ReLU() ]
            layers += [ nn.Linear(c.internal_width,  c_out) ]
            return nn.Sequential(*layers)
        
        nodes = [Ff.InputNode(c.y_dim_features, name='input')]
        for i in range(c.n_blocks):
            nodes.append(Ff.Node([nodes[-1].out0], Fm.GLOWCouplingBlock,
                              {'subnet_constructor':fc_constr,
                               'clamp':c.exponent_clamping},
                              name=F'coubpling_{i}'))
            if c.use_permutation:
                nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {'seed':i}, name=F'permute_{i}'))
        nodes.append(Ff.OutputNode([nodes[-1].out0], name='output'))
       
        model = INN(nodes, verbose=False)
        model.to(c.device)
        
        return model
    
    def optim_step(self):
        self.optim.step()
        self.optim.zero_grad()
    
    def scheduler_step(self):
        self.weight_scheduler.step()
        
    def save(self, name):
        torch.save({'opt':self.optim.state_dict(),
                    'net':self.model.state_dict(),}, name)
        
    def load(self, name, device='cpu'):
        state_dicts = torch.load(name, map_location=torch.device(device), weights_only=False)
        self.model.load_state_dict(state_dicts['net'])
        try:
            self.optim.load_state_dict(state_dicts['opt'])
        except ValueError:
            print('Cannot load optimizer for some reason or other')
       
    
    def forward(self, x):
        return self.model(x)



# nodes = [ Ff.InputNode(c.y_dim_features, name='input')]
# nodes.append( Ff.OutputNode( nodes[-1].out0, name='output') )
 
