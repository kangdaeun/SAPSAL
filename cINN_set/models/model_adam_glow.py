# 2022. 1. 10. Ver007 FrEIA version 0.1, 0.2
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

# from FrEIA.framework import *
# from FrEIA.modules import *
#import FrEIA0p1.framework as Ff0p1
#import FrEIA0p1.modules as Fm0p1
#import FrEIA0p2.framework as Ff0p2
#import FrEIA0p2.modules as Fm0p2
from ..FrEIA import framework as Ff
from ..FrEIA import modules as Fm


class FeatureNet(nn.Module):
    def __init__(self, c):
        super().__init__()

        # self.linear = nn.Sequential(
        #             nn.Linear(c.y_dim_in, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, 512),
        #             nn.ReLU(),
        #             nn.Linear(512, c.y_dim_features),
        #             )
        layers = [nn.Linear(c.y_dim_in, c.feature_width), nn.ReLU()]
        for i_layer in range(1, c.feature_layer-1):
            layers += [ nn.Linear(c.feature_width, c.feature_width), nn.ReLU()]
        layers += [ nn.Linear(c.feature_width, c.y_dim_features) ]
        self.linear = nn.Sequential(*layers)

        self.fc_final = nn.Linear(c.y_dim_features, c.x_dim)

    def forward(self, x):
        x = self.linear(x)
        return self.fc_final(x)

    def features(self, x):
        return self.linear(x)


class ModelAdamGLOW(nn.Module):
    
    def __init__(self, c):
        
        super().__init__()
        
        self.cond_net = FeatureNet(c)
        self.cond_net.to(c.device)
        self.model = self.build_network(c)
        
        self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        torch.manual_seed(c.seed_weight_init)
        for p in self.params_trainable:
             p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)
        self.params_trainable += list(self.cond_net.parameters())
        
        self.optim = torch.optim.Adam(self.params_trainable, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
        self.weight_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=c.meta_epoch, gamma=c.gamma)
        
        # save parameters for rescaling in dictionary (mu_x, mu_y, w_x, w_y)
        self.rescale_params = None
    
    def build_network(self, c):
        
#        if c.FrEIA_ver == 0.1:
#            Ff = Ff0p1; Fm = Fm0p1
#            INN = Ff.ReversibleGraphNet
#        elif c.FrEIA_ver == 0.2:
#            Ff = Ff0p2; Fm = Fm0p2
#            INN = Ff.GraphINN
#        else: # 2022.1.10. currently using 0.1 as default
#            Ff = Ff0p1; Fm = Fm0p1
#            INN = Ff.ReversibleGraphNet
        # 2023. 8. 11. FrEIA_ver deprecated (use FrEIA >= 0.2 in cINN_set)
        INN = Ff.GraphINN
        
        # def fc_constr(c_in, c_out):
        #     return nn.Sequential(nn.Linear(c_in, c.internal_width), nn.ReLU(),
        #                          nn.Linear(c.internal_width,  c.internal_width), nn.ReLU(),
        #                          nn.Linear(c.internal_width,  c_out))
        
        def fc_constr(c_in, c_out):
            layers = [ nn.Linear(c_in, c.internal_width), nn.ReLU() ]
            for i_layer in range(1, c.internal_layer-1):
                layers += [ nn.Linear(c.internal_width,  c.internal_width), nn.ReLU() ]
            layers += [ nn.Linear(c.internal_width,  c_out) ]
            return nn.Sequential(*layers)
            
        
        nodes = [Ff.ConditionNode(c.y_dim_features, name='cond'), Ff.InputNode(c.x_dim, name='input')]
        for i in range(c.n_blocks):
            nodes.append(Ff.Node([nodes[-1].out0], Fm.GLOWCouplingBlock,
                              {'subnet_constructor':fc_constr,
                               'clamp':c.exponent_clamping},
                              conditions = [nodes[0]],
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
                    'net':self.model.state_dict(),
                    'cond_net':self.cond_net.state_dict(),
                    'rescale_params':self.rescale_params}, name)

    def load(self, name, device='cpu'):
        state_dicts = torch.load(name, map_location=torch.device(device))
        self.model.load_state_dict(state_dicts['net'])
        self.cond_net.load_state_dict(state_dicts['cond_net'])
        try:
            self.optim.load_state_dict(state_dicts['opt'])
        except ValueError:
            print('Cannot load optimizer for some reason or other')
            
        try:
            self.rescale_params = state_dicts['rescale_params']
        except KeyError:
            print('Rescale parameter dictionary is not saved in the model file')
       
    
