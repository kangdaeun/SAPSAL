from scipy.ndimage import zoom
import numpy as np
import os
import sys
import importlib.util
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

     
class Visualizer:
    def __init__(self, loss_labels):
            self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels
            self.counter = 0

            header = 'Epoch'
            for l in loss_labels:
                header += '\t\t%s' % (l)

            print(header)

    def update_losses(self, losses, *args, **kwargs):
        print('\r', '    '*20, end='')
        line = '\r%.3i' % (self.counter)
        for l in losses:
            line += '\t\t%.4f' % (l)

        print(line, flush=True)
        self.counter += 1

    def update_images(self, *args):
        pass

    def update_hist(self, *args):
        pass
    
    
class TrainVisualizer():
    
    def __init__(self, c):
        
        if c.live_visualization:
            self.visualizer = self.build_liveviz(c)
        else:
            self.visualizer = Visualizer(c.loss_names)
        
        
    def build_liveviz(self, c):
        
        import visdom
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
        n_plots = 2
        figsize = (4,4)
    
        class LiveVisualizer(Visualizer):
            def __init__(self, loss_labels):
                super().__init__(loss_labels)
                self.viz = visdom.Visdom()
                self.viz.close()
    
                self.l_plots = self.viz.line(X = np.zeros((1,self.n_losses)),
                                             Y = np.zeros((1,self.n_losses)),
                                             opts = {'legend':self.loss_labels})
    
                self.fig, self.axes = plt.subplots(n_plots, n_plots, figsize=figsize)
                self.hist = self.viz.matplot(self.fig)
    
    
            def update_losses(self, losses, its=None, logscale=True):
                super().update_losses(losses)
                # its = min(len(c.train_loader), c.n_its_per_epoch)
                y = np.array([losses])
                if logscale:
                    y = np.log10(y)
    
                self.viz.line(X = (self.counter-1) * its * np.ones((1,self.n_losses)),
                              Y = y,
                              opts   = {'legend':self.loss_labels},
                              win    = self.l_plots,
                              update = 'append')
    
            def update_hist(self, data):
                for i in range(n_plots):
                    for j in range(n_plots):
                        try:
                            self.axes[i,j].clear()
                            self.axes[i,j].hist(data[:, i*n_plots + j], bins=20, histtype='step')
                        except ValueError:
                            pass
    
                self.fig.tight_layout()
                self.viz.matplot(self.fig, win=self.hist)
    
            def close(self):
                self.viz.close(win=self.hist)
                self.viz.close(win=self.l_plots)
    
        return LiveVisualizer(c.loss_names)
        

    def show_loss(self, losses, its=None, logscale=True):
        self.visualizer.update_losses(losses, its=its, logscale=logscale)
    
    def show_hist(self, data):
        self.visualizer.update_hist(data.data)
    
    def close(self):
        self.visualizer.close()
    
    
# Plot

def make_txt_info(c, title=None, title_append=True):
    
    infos = {
                'model_code': c.model_code,
                'adam_betas': 'Adb: (%.2g,%.2g)'%(c.adam_betas),
                'batch_size': 'B: %d'%int(c.batch_size),
                'n_blocks': '$N_{\mathrm{block}}$: %d'%int(c.n_blocks),
                'gamma': '$\gamma_{\mathrm{decay}}$: %.3g'%c.gamma,
                'lr_init': '$Lr_{\mathrm{init}}$: %.3g'%c.lr_init,
                'l2_weight_reg': '$L2_{\mathrm{reg}}$: %.1e'%c.l2_weight_reg,
                'meta_epoch': '$Sc_{\mathrm{epoch}}$: %d'%int(c.meta_epoch), 
                'test_frac': '$f_{\mathrm{test}}$: %.2g'%(c.test_frac),
                'internal_layer': '$N_{\mathrm{sub,layer}}$: %d'%int(c.internal_layer),
                            }
        
    txt = []
    txt.append(infos['model_code'])
    if title is None:
        title = r'%s, %s, %s'%(infos['batch_size'],infos['lr_init'], infos['gamma'])
    elif title_append:
        title += '\t'+r'(%s, %s, %s)'%(infos['batch_size'],infos['lr_init'], infos['gamma'])
    else:
        txt += [infos['batch_size'],infos['lr_init'], infos['gamma']]

    txt += [infos['l2_weight_reg'], infos['adam_betas'], infos['meta_epoch'], 
            infos['test_frac'], infos['n_blocks'], infos['internal_layer']]
    
    return txt, title
    

def plot_loss_curve(epoch, loss, test_loss, c=None, figname=None, figsize=[5, 4.5], resi_bound=5, 
              yrange=None, xrange=None, title=None, title_append=True, titlesize='medium', ticklabelsize='large', xylabelsize='x-large', grid=True):
    
    import matplotlib
    matplotlib.use('Agg')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
#    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=figsize)
    l0=ax.plot(epoch, loss, linewidth=2, label='Train (%.2f)'%(np.mean(loss[-6:-1])))
    l1=ax.plot(epoch, test_loss, '--', label='Test (%.2f)'%(np.mean(test_loss[-6:-1])))
    
#     ax.set_xlabel('Epoch', fontsize=xylabelsize)
    ax.set_ylabel('Loss', fontsize=xylabelsize)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='both',top='on',right='on',direction='in' )
    ax.tick_params(axis='both',which='major', labelsize=ticklabelsize )
    ax.tick_params(axis='x',labelbottom=False)
    if grid:
        ax.grid(alpha=0.5)
      
        
    if c is not None:
        txt, title = make_txt_info(c, title=title, title_append=title_append)
            
        ax.text(0.96,0.96, '\n'.join(txt), ha='right', va='top', transform=ax.transAxes, fontsize=titlesize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))
    
    
    ax.set_title(title, fontsize=titlesize)
    
    if yrange is not None:
        ax.set_ylim(yrange)
    else:
        if (ax.get_ylim()[1] - ax.get_ylim()[0]) > 1000:
            ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+1000 ])
    
    
    if xrange is not None:
        ax.set_xlim(xrange)
        
    divider = make_axes_locatable(ax)
    resi_ax = divider.append_axes("bottom",size="20%", pad="5%")
    resi = test_loss - loss
    resi_ax.plot(epoch, resi, color=l1[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
    resi_ax.axhline(y=0, ls='--', color=l0[0].get_color())
    resi_ax.set_xlim(ax.get_xlim())
    yr = list(resi_ax.get_ylim())
    
    if abs(yr[0]) >= resi_bound:
        yr[0] = -resi_bound
    if abs(yr[1]) >= resi_bound:
        yr[1] = resi_bound
    resi_ax.set_ylim(yr)
    resi_ax.set_xlabel('Epoch', fontsize=xylabelsize)
    resi_ax.set_ylabel('Res', fontsize=xylabelsize)
    if grid:
        resi_ax.grid(alpha=0.5)
    
    ax.legend(loc='upper left')
    resi_ax.legend(loc='lower right')
        
    fig.tight_layout()
    if figname:
        fig.savefig(figname,dpi=250)
        plt.close()
    
    return fig, ax

def plot_loss_curve_2types(epoch, loss_mn, test_loss_mn, loss_mdn, test_loss_mdn, 
                           c=None, figname=None, figsize=[10, 5], resi_bound=5, 
              yrange=None, xrange=None, title=None, title_append=True, titlesize='large', ticklabelsize='large', xylabelsize='x-large', grid=True):
    
    import matplotlib
    matplotlib.use('Agg')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
#    matplotlib.use('Agg')
    fig, axis = plt.subplots(1,2,figsize=figsize)
    
    for i_type in range(2):
        if i_type == 0:
            loss = loss_mn
            test_loss = test_loss_mn
            ylabel = 'Loss (mean)'
        else:
            loss = loss_mdn
            test_loss = test_loss_mdn
            ylabel = 'Loss (median)'
    
    
        ax = axis[i_type]
        l0=ax.plot(epoch, loss, linewidth=2, label='Train (%.2f)'%(np.mean(loss[-6:-1])))
        l1=ax.plot(epoch, test_loss, '--', label='Test (%.2f)'%(np.mean(test_loss[-6:-1])))

    #     ax.set_xlabel('Epoch', fontsize=xylabelsize)
        ax.set_ylabel(ylabel, fontsize=xylabelsize)
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',top='on',right='on',direction='in' )
        ax.tick_params(axis='both',which='major', labelsize=ticklabelsize )
        ax.tick_params(axis='x',labelbottom=False)
        if grid:
            ax.grid(alpha=0.5)

        if yrange is not None:
            ax.set_ylim(yrange)
        else:
            if (ax.get_ylim()[1] - ax.get_ylim()[0]) > 1000:
                ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+1000 ])

        if xrange is not None:
            ax.set_xlim(xrange)
        
        divider = make_axes_locatable(ax)
        resi_ax = divider.append_axes("bottom",size="20%", pad="5%")
        resi = test_loss - loss
        resi_ax.plot(epoch, resi, color=l1[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi_ax.axhline(y=0, ls='--', color=l0[0].get_color())
        resi_ax.set_xlim(ax.get_xlim())
        yr = list(resi_ax.get_ylim())

        if abs(yr[0]) >= resi_bound:
            yr[0] = -resi_bound
        if abs(yr[1]) >= resi_bound:
            yr[1] = resi_bound
        resi_ax.set_ylim(yr)
        resi_ax.set_xlabel('Epoch', fontsize=xylabelsize)
        resi_ax.set_ylabel('Res', fontsize=xylabelsize)
        if grid:
            resi_ax.grid(alpha=0.5)

        ax.legend(loc='upper left')
        resi_ax.legend(loc='lower right')
    
    
    
    if c is not None:
        
        txt, title = make_txt_info(c, title=title, title_append=title_append)
        ax.text(0.96,0.96, '\n'.join(txt), ha='right', va='top', transform=ax.transAxes, fontsize=titlesize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))
    
    
    fig.suptitle(title, fontsize=titlesize)
    
    
        
    fig.tight_layout()
    if figname:
        fig.savefig(figname,dpi=250)
        plt.close()
    
    return fig, ax
