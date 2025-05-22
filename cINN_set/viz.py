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
                'n_blocks': r'$N_{\mathrm{block}}$: %d'%int(c.n_blocks),
                'gamma': r'$\gamma_{\mathrm{decay}}$: %.3g'%c.gamma,
                'lr_init': r'$Lr_{\mathrm{init}}$: %.2e'%c.lr_init,
                'l2_weight_reg': r'$L2_{\mathrm{reg}}$: %.1e'%c.l2_weight_reg,
                'meta_epoch': r'$Sc_{\mathrm{epoch}}$: %d'%int(c.meta_epoch), 
                'test_frac': r'$f_{\mathrm{test}}$: %.2g'%(c.test_frac),
                'internal_layer': r'$N_{\mathrm{sub,layer}}$: %d'%int(c.internal_layer),
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

def make_txt_info_DA(c):

    # dtype fixed parameters
    infos = {
                'lam_adv': r'$\lambda_{\mathrm{adv}}$: %.3g'%(c.lambda_adv),
                
                'disc_width': r'$N_{\mathrm{Disc,width}}$: %d'%int(c.da_disc_width),
                'disc_layer': r'$N_{\mathrm{Disc,layer}}$: %d'%int(c.da_disc_layer),
                
                'gamma': r'$\gamma_{\mathrm{Disc}}$: %.3g'%c.da_disc_gamma,
                'lr_init': r'$Lr_{\mathrm{init,D}}$: %.3g'%c.da_disc_lr_init,
                'l2_weight_reg': r'$L2_{\mathrm{reg,D}}$: %.1e'%c.da_disc_l2_weight_reg,
                'meta_epoch': r'$Sc_{\mathrm{epoch},D}$: %d'%int(c.da_disc_meta_epoch), 
                'adam_betas': 'Adb(D): (%.2g,%.2g)'%(c.da_disc_adam_betas),
                'real_frac': r'$f_{\mathrm{real}}$: %.2g'%(c.real_frac),
               
                'delay_cinn': r'$N_{\mathrm{delay}}$: %d'%c.delay_cinn,
                'delay_disc': r'$N_{\mathrm{delay,D}}$: %d'%c.delay_disc,
                'n_warmup': r'$N_{\mathrm{warmup}}$: %d'%c.da_warmup,
                'disc_warmup_delay': r'$N_{\mathrm{wu,delay}}$: %d'%c.da_disc_warmup_delay,
   
                'label_smoothing': 'LSmooth: %s'%str(c.da_label_smoothing),
                
                            }
    if c.da_disc_train_step is None:
        infos['train_step']=r'$Ts_{\mathrm{epoch}}$: auto'
    elif type(c.da_disc_train_step)==int:
        infos['train_step']=r'$Ts_{\mathrm{epoch}}$: %d'%int(c.da_disc_train_step)
    else: 
        infos['train_step']=r'$Ts_{\mathrm{epoch}}$: %s'%str(c.da_disc_train_step)

    
    # pile up 
    txt = []
    if c.da_mode=='simple':
        txt.append(infos['lam_adv'])
    txt += [ infos['real_frac'], infos['train_step'], ]
    
    if c.da_label_smoothing:
        txt.append( infos['label_smoothing'])

    if c.da_warmup > 0:
        txt.append( infos['n_warmup'])
    if c.da_disc_warmup_delay > 0:
        txt.append( infos['disc_warmup_delay'])
    if c.delay_cinn > 0:
        txt.append( infos['delay_cinn'])
    if c.delay_disc > 0:
        txt.append( infos['delay_disc'])
    if c.da_stop_da is not None:
        infos['da_stop'] = r'$N_{\mathrm{DAstop}}$: %d'%c.da_stop_da
        txt.append( infos['da_stop'])

    # learning and size
    txt += [ infos['lr_init'],infos['gamma'],
                infos['l2_weight_reg'], infos['adam_betas'], infos['meta_epoch']]
 
    txt += [infos['disc_width'], infos['disc_layer']]
    
    return txt
    
    

def plot_loss_curve(epoch, loss, test_loss, c=None, figname=None, figsize=[5, 4.5], resi_bound=5, 
              yrange=None, xrange=None, title=None, title_append=True, titlesize='medium', ticklabelsize='large', xylabelsize='x-large', grid=True):
    
    
    matplotlib.use('Agg')
  
    fig, ax = plt.subplots(figsize=figsize)
    l0=ax.plot(epoch, loss, linewidth=2, label='Train (%.2f)'%(np.mean(loss[-6:-1])))
    l1=ax.plot(epoch, test_loss, '--', label='Test (%.2f)'%(np.mean(test_loss[-6:-1])))
    

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
    resi_ax.axhline(y=0, ls='--', color='k')
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


def plot_loss_curve_2types(epoch_loss_history,
                              c=None, figname=None, figsize=[10, 5], resi_bound=5, 
                              yrange=None, xrange=None, title=None, title_append=True, 
                              titlesize='large', ticklabelsize='large', xylabelsize='x-large', grid=True,
                              return_figure=False,
                              ):
    
    import matplotlib
    matplotlib.use('Agg')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # draw values
    epoch = epoch_loss_history['Epoch']
    loss_mn = epoch_loss_history['Loss_train_mn']
    test_loss_mn = epoch_loss_history['Loss_test_mn']
    loss_mdn = epoch_loss_history['Loss_train_mdn']
    test_loss_mdn = epoch_loss_history['Loss_test_mdn']

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
        resi_ax.axhline(y=0, ls='--', color='k')
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
    
    if return_figure:
        return fig, ax

def plot_loss_curve_DA(epoch_loss_history,
                           c=None, figname=None, figsize=[13, 7.5], resi_bound=5, 
                          yrange=None, xrange=None, title=None, title_append=True, 
                           titlesize='large', ticklabelsize='large', xylabelsize='large', 
                       txtsize='small', legsize='small', grid=True,
                       return_figure=False,
                       
                       ):
    

    matplotlib.use('Agg')
  
    # if c.da_mode=='simple': 
    #     figsize = [13, 8]
    #     ncol = 3
    # elif c.da_mode =='WGAN':
    #     figsize = [8.2,8]
    #     ncol = 2
    # else:
    #     ncol = 2
  
    fig, axis = plt.subplots(2, 3,figsize=figsize, tight_layout=True)
    epoch = epoch_loss_history['Epoch']

    # (Ltot, Lcinn), (Ldisc, Ladv), (feature distance)
    
    for i_type in range(2):
        if i_type==0: # use median
            loss_tot = epoch_loss_history['Loss_train_mdn']
            test_loss_tot = epoch_loss_history['Loss_test_mdn']
            
            loss_cinn = epoch_loss_history['NLLLoss_train_mdn']
            test_loss_cinn = epoch_loss_history['NLLLoss_test_mdn']

            loss_d = epoch_loss_history['DLoss_train_mdn']
            test_loss_d = epoch_loss_history['DLoss_test_mdn']
            
            loss_adv = epoch_loss_history['AdvLoss_train_mdn']
            test_loss_adv = epoch_loss_history['AdvLoss_test_mdn']

            feat_dist = epoch_loss_history['Feature_Dist']
                
            ylabel_adv = r'L$_{\mathrm{Adv}}$'
            ylabel_gen = 'Loss (median)'
            ylabel_tot = r'L$_{\mathrm{Tot}}$'
            ylabel_nll = r'L$_{\mathrm{NLL}}$'
            ylabel_disc = r'L$_{\mathrm{Disc}}$'

            ylabel_fd = 'Feature distance'
        elif i_type == 1: # use mean
            loss_tot = epoch_loss_history['Loss_train_mn']
            test_loss_tot = epoch_loss_history['Loss_test_mn']
    
            loss_cinn = epoch_loss_history['NLLLoss_train_mn']
            test_loss_cinn = epoch_loss_history['NLLLoss_test_mn']

            loss_d = epoch_loss_history['DLoss_train_mn']
            test_loss_d = epoch_loss_history['DLoss_test_mn']
            loss_adv = epoch_loss_history['AdvLoss_train_mn']
            test_loss_adv = epoch_loss_history['AdvLoss_test_mn']
            
            ylabel_adv = r'L$_{\mathrm{Adv}}$'
            ylabel_gen = 'Loss (mean)'
            ylabel_tot = r'L$_{\mathrm{Tot}}$'
            ylabel_nll = r'L$_{\mathrm{NLL}}$'
            ylabel_disc = r'L$_{\mathrm{Disc}}$'
        
    

        # total loss & NLL loss (cINN)
        ax = axis[i_type, 0]
        l0=ax.plot(epoch, loss_tot, linewidth=2, label='Train %s (%.2f)'%(ylabel_tot,np.mean(loss_tot[-6:-1])))
        l1=ax.plot(epoch, test_loss_tot, '--', label='Test %s (%.2f)'%(ylabel_tot, np.mean(test_loss_tot[-6:-1])))
        l2=ax.plot(epoch, loss_cinn, linewidth=2, label='Train %s (%.2f)'%(ylabel_nll, np.mean(loss_cinn[-6:-1])))
        l3=ax.plot(epoch, test_loss_cinn, '--', label='Test %s (%.2f)'%(ylabel_nll, np.mean(test_loss_cinn[-6:-1])))
        # ax.set_xlabel('Epoch', fontsize=xylabelsize)
        ax.set_ylabel(ylabel_gen, fontsize=xylabelsize)

        if yrange is not None:
            ax.set_ylim(yrange)
        else:
            if (ax.get_ylim()[1] - ax.get_ylim()[0]) > 1000:
                ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+1000 ])

        if xrange is not None:
            ax.set_xlim(xrange)
        
        divider = make_axes_locatable(ax)
        resi_ax = divider.append_axes("bottom",size="20%", pad="3%")
        resi = test_loss_tot - loss_tot
        resi_ax.plot(epoch, resi, color=l1[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi = test_loss_cinn - loss_cinn
        resi_ax.plot(epoch, resi, color=l3[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi_ax.axhline(y=0, ls='--', color='k')
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
        ax.legend(loc='upper left', fontsize=legsize)
        resi_ax.legend(loc='lower right',fontsize=legsize)

    

        # Discriminator loss, adversarial loss
        ax = axis[i_type, 1]
        l0=ax.plot(epoch, loss_d, linewidth=2, label='Train %s (%.2f)'%(ylabel_disc, np.mean(loss_d[-6:-1])))
        l1=ax.plot(epoch, test_loss_d, '--', label='Test %s (%.2f)'%(ylabel_disc, np.mean(test_loss_d[-6:-1])))
        l2=ax.plot(epoch, loss_adv, linewidth=2, label='Train %s (%.2f)'%(ylabel_adv, np.mean(loss_adv[-6:-1])))
        l3=ax.plot(epoch, test_loss_adv, '--', label='Test %s (%.2f)'%(ylabel_adv, np.mean(test_loss_adv[-6:-1])))
        ax.set_ylabel(ylabel_gen, fontsize=xylabelsize)

        if (ax.get_ylim()[1] - ax.get_ylim()[0]) > 1000:
            ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+1000 ])

        da_range = [0, epoch[-1]]
        if c.da_stop_da is not None:
            da_end = c.da_warmup + c.da_stop_da +1
            if da_range[1] > da_end: da_range[1] = da_end
        ax.set_xlim(da_range)

        divider = make_axes_locatable(ax)
        resi_ax = divider.append_axes("bottom",size="20%", pad="3%")
        resi = test_loss_d - loss_d
        resi_ax.plot(epoch, resi, color=l1[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi = test_loss_adv - loss_adv
        resi_ax.plot(epoch, resi, color=l3[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi_ax.axhline(y=0, ls='--', color='k')
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
        ax.legend(loc='upper left',fontsize=legsize)
        resi_ax.legend(loc='lower right',fontsize=legsize)
        
        
        # 3 feature distance
        if i_type==0:
            ax = axis[i_type, 2]
            l0=ax.plot(epoch, feat_dist, linewidth=2, label='Test (%.2f)'%( np.mean(feat_dist[-6:-1])))
            ax.set_ylabel(ylabel_fd, fontsize=xylabelsize)
    
            if c.da_warmup > 0:
                ax.axvline(x=c.da_warmup, color='C1', ls='--', label='Warm-up')
            if c.da_stop_da is not None:
                da_end = c.da_warmup + c.da_stop_da
                ax.axvline(x=da_end, color='C1', ls='-.', label='DA stop')
           
            ax.set_xlim(da_range)
            ax.legend(loc='upper left',fontsize=legsize)
            ax.set_xlabel('Epoch', fontsize=xylabelsize)
        else:
            pass
      
    # common tick setup for all axsis
    for ax in axis.ravel():
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',top='on',right='on',direction='in' )
        ax.tick_params(axis='both',which='major', labelsize=ticklabelsize )
        ax.tick_params(axis='x',labelbottom=False)
        if grid:
            ax.grid(alpha=0.5)
    
    axis[1,2].axis('off')
    axis[0,2].tick_params(axis='x',labelbottom=True)
    if c is not None:
        txt, title = make_txt_info(c, title=title, title_append=title_append)
        axis[0,0].text(0.96,0.96, '\n'.join(txt), ha='right', va='top', transform=axis[0,0].transAxes, fontsize=txtsize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))

        txt = make_txt_info_DA(c)
        
        nn = int(np.ceil(len(txt)*0.5))
        txt1 = txt[:nn]
        txt2 = txt[nn:]
        axis[0,1].text(0.96,0.96, '\n'.join(txt1), ha='right', va='top', transform=axis[0,1].transAxes, fontsize=txtsize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))
        axis[0,2].text(0.96,0.96, '\n'.join(txt2), ha='right', va='top', transform=axis[0,2].transAxes, fontsize=txtsize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))
        
    
    fig.suptitle(title, fontsize=titlesize)
    fig.tight_layout()
    if figname:
        fig.savefig(figname,dpi=250)
        plt.close()
   
    if return_figure:
        return fig, axis
    
    
def plot_loss_curve_DAwoD(epoch_loss_history,
                           c=None, figname=None, figsize=[8, 7.5], resi_bound=5, 
                          yrange=None, xrange=None, title=None, title_append=True, 
                           titlesize='large', ticklabelsize='large', xylabelsize='large', 
                       txtsize='small', legsize='small', grid=True,
                       return_figure=False,
                       
                       ):
    

    matplotlib.use('Agg')
  
    
    fig, axis = plt.subplots(2, 2,figsize=figsize, tight_layout=True)
    epoch = epoch_loss_history['Epoch']

    # (Ltot, Lcinn), (Ldisc, Ladv), (feature distance)
    
    for i_type in range(2):
        if i_type==0: # use median
            loss_tot = epoch_loss_history['Loss_train_mdn']
            test_loss_tot = epoch_loss_history['Loss_test_mdn']
            
            loss_cinn = epoch_loss_history['NLLLoss_train_mdn']
            test_loss_cinn = epoch_loss_history['NLLLoss_test_mdn']

            feat_dist = epoch_loss_history['Feature_Dist']
                
           
            ylabel_gen = 'Loss (median)'
            ylabel_tot = r'L$_{\mathrm{Tot}}$'
            ylabel_nll = r'L$_{\mathrm{NLL}}$'
    

            ylabel_fd = 'Feature distance'
        elif i_type == 1: # use mean
            loss_tot = epoch_loss_history['Loss_train_mn']
            test_loss_tot = epoch_loss_history['Loss_test_mn']
    
            loss_cinn = epoch_loss_history['NLLLoss_train_mn']
            test_loss_cinn = epoch_loss_history['NLLLoss_test_mn']


            ylabel_gen = 'Loss (mean)'
            ylabel_tot = r'L$_{\mathrm{Tot}}$'
            ylabel_nll = r'L$_{\mathrm{NLL}}$'
           
        
    

        # total loss & NLL loss (cINN)
        ax = axis[i_type, 0]
        l0=ax.plot(epoch, loss_tot, linewidth=2, label='Train %s (%.2f)'%(ylabel_tot,np.mean(loss_tot[-6:-1])))
        l1=ax.plot(epoch, test_loss_tot, '--', label='Test %s (%.2f)'%(ylabel_tot, np.mean(test_loss_tot[-6:-1])))
        l2=ax.plot(epoch, loss_cinn, linewidth=2, label='Train %s (%.2f)'%(ylabel_nll, np.mean(loss_cinn[-6:-1])))
        l3=ax.plot(epoch, test_loss_cinn, '--', label='Test %s (%.2f)'%(ylabel_nll, np.mean(test_loss_cinn[-6:-1])))
        # ax.set_xlabel('Epoch', fontsize=xylabelsize)
        ax.set_ylabel(ylabel_gen, fontsize=xylabelsize)

        if yrange is not None:
            ax.set_ylim(yrange)
        else:
            if (ax.get_ylim()[1] - ax.get_ylim()[0]) > 1000:
                ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+1000 ])

        if xrange is not None:
            ax.set_xlim(xrange)
        
        divider = make_axes_locatable(ax)
        resi_ax = divider.append_axes("bottom",size="20%", pad="3%")
        resi = test_loss_tot - loss_tot
        resi_ax.plot(epoch, resi, color=l1[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi = test_loss_cinn - loss_cinn
        resi_ax.plot(epoch, resi, color=l3[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi_ax.axhline(y=0, ls='--', color='k')
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
        ax.legend(loc='upper left', fontsize=legsize)
        resi_ax.legend(loc='lower right',fontsize=legsize)

    

        # 2 feature distance
        if i_type==0:
            ax = axis[i_type, 1]
            l0=ax.plot(epoch, feat_dist, linewidth=2, label='Test (%.2f)'%( np.mean(feat_dist[-6:-1])))
            ax.set_ylabel(ylabel_fd, fontsize=xylabelsize)
    
            if xrange is not None:
                ax.set_xlim(xrange)
            ax.legend(loc='upper left',fontsize=legsize)
            ax.set_xlabel('Epoch', fontsize=xylabelsize)
        else:
            pass
      
    # common tick setup for all axsis
    for ax in axis.ravel():
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',top='on',right='on',direction='in' )
        ax.tick_params(axis='both',which='major', labelsize=ticklabelsize )
        ax.tick_params(axis='x',labelbottom=False)
        if grid:
            ax.grid(alpha=0.5)
    
    axis[1,1].axis('off')
    axis[0,1].tick_params(axis='x',labelbottom=True)
    if c is not None:
        txt, title = make_txt_info(c, title=title, title_append=title_append)
        axis[0,0].text(0.96,0.96, '\n'.join(txt), ha='right', va='top', transform=axis[0,0].transAxes, fontsize=txtsize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))
        
    
    fig.suptitle(title, fontsize=titlesize)
    fig.tight_layout()
    if figname:
        fig.savefig(figname,dpi=250)
        plt.close()
   
    if return_figure:
        return fig, axis


def _plot_loss_curve_DA(epoch_loss_history,
                           c=None, figname=None, figsize=[13, 8], resi_bound=5, 
                          yrange=None, xrange=None, title=None, title_append=True, 
                           titlesize='large', ticklabelsize='large', xylabelsize='large', 
                       txtsize='small', legsize='small',
                       grid=True):
    

    matplotlib.use('Agg')
  

    fig, axis = plt.subplots(2,3,figsize=figsize, tight_layout=True)
    epoch = epoch_loss_history['Epoch']
    
    for i_type in range(2):
        if i_type == 1:
            loss_tot = epoch_loss_history['Loss_train_mn']
            test_loss_tot = epoch_loss_history['Loss_test_mn']
            
            loss_cinn = epoch_loss_history['NLLLoss_train_mn']
            test_loss_cinn = epoch_loss_history['NLLLoss_test_mn']
            loss_adv = epoch_loss_history['AdvLoss_train_mn']
            test_loss_adv = epoch_loss_history['AdvLoss_test_mn']
            loss_d = epoch_loss_history['DLoss_train_mn']
            test_loss_d = epoch_loss_history['DLoss_test_mn']
            
            ylabel1 = 'Loss (mean)'
            ylabel11 = r'L$_{\mathrm{Tot}}$'
            ylabel12 = r'L$_{\mathrm{NLL}}$'
            ylabel2 = r'L$_{\mathrm{Adv}}$ (mean)'
            ylabel3 = r'L$_{\mathrm{Disc}}$ (mean)'
        else:
            loss_tot = epoch_loss_history['Loss_train_mdn']
            test_loss_tot = epoch_loss_history['Loss_test_mdn']
            
            loss_cinn = epoch_loss_history['NLLLoss_train_mdn']
            test_loss_cinn = epoch_loss_history['NLLLoss_test_mdn']
            loss_adv = epoch_loss_history['AdvLoss_train_mdn']
            test_loss_adv = epoch_loss_history['AdvLoss_test_mdn']
            loss_d = epoch_loss_history['DLoss_train_mdn']
            test_loss_d = epoch_loss_history['DLoss_test_mdn']
            
            ylabel1 = 'Loss (median)'
            ylabel11 = r'L$_{\mathrm{Tot}}$'
            ylabel12 = r'L$_{\mathrm{NLL}}$'
            ylabel2 = r'L$_{\mathrm{Adv}}$ (median)'
            ylabel3 = r'L$_{\mathrm{Disc}}$ (median)'
    

        # total loss & NLL loss (cINN)
        ax = axis[i_type, 0]
        l0=ax.plot(epoch, loss_tot, linewidth=2, label='Train %s (%.2f)'%(ylabel11,np.mean(loss_tot[-6:-1])))
        l1=ax.plot(epoch, test_loss_tot, '--', label='Test %s (%.2f)'%(ylabel11, np.mean(test_loss_tot[-6:-1])))
        l2=ax.plot(epoch, loss_cinn, linewidth=2, label='Train %s (%.2f)'%(ylabel12, np.mean(loss_cinn[-6:-1])))
        l3=ax.plot(epoch, test_loss_cinn, '--', label='Test %s (%.2f)'%(ylabel12, np.mean(test_loss_cinn[-6:-1])))
        # ax.set_xlabel('Epoch', fontsize=xylabelsize)
        ax.set_ylabel(ylabel1, fontsize=xylabelsize)

        if yrange is not None:
            ax.set_ylim(yrange)
        else:
            if (ax.get_ylim()[1] - ax.get_ylim()[0]) > 1000:
                ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+1000 ])

        if xrange is not None:
            ax.set_xlim(xrange)
        
        divider = make_axes_locatable(ax)
        resi_ax = divider.append_axes("bottom",size="20%", pad="3%")
        resi = test_loss_tot - loss_tot
        resi_ax.plot(epoch, resi, color=l1[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi = test_loss_cinn - loss_cinn
        resi_ax.plot(epoch, resi, color=l3[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi_ax.axhline(y=0, ls='--', color='k')
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
        ax.legend(loc='upper left', fontsize=legsize)
        resi_ax.legend(loc='lower right',fontsize=legsize)

        
        # 2.Adversari loss 
        ax = axis[i_type, 1]
    
        l0=ax.plot(epoch, loss_adv, linewidth=2, label='Train (%.2f)'%( np.mean(loss_adv[-6:-1])))
        l1=ax.plot(epoch, test_loss_adv, '--', label='Test (%.2f)'%( np.mean(test_loss_adv[-6:-1])))
        # ax.set_xlabel('Epoch', fontsize=xylabelsize)
        ax.set_ylabel(ylabel2, fontsize=xylabelsize)
        
        # if yrange is not None:
        #     ax.set_ylim(yrange)
        # else:
        if (ax.get_ylim()[1] - ax.get_ylim()[0]) > 1000:
            ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+1000 ])
        if xrange is not None:
            ax.set_xlim(xrange)
        
        divider = make_axes_locatable(ax)
        resi_ax = divider.append_axes("bottom",size="20%", pad="3%")
        resi = test_loss_adv - loss_adv
        resi_ax.plot(epoch, resi, color=l1[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi_ax.axhline(y=0, ls='--', color='k')
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
        ax.legend(loc='upper left',fontsize=legsize)
        resi_ax.legend(loc='lower right',fontsize=legsize)

    # 3. Discriminator loss
        ax = axis[i_type, 2]
        l0=ax.plot(epoch, loss_d, linewidth=2, label='Train (%.2f)'%( np.mean(loss_d[-6:-1])))
        l1=ax.plot(epoch, test_loss_d, '--', label='Test (%.2f)'%(np.mean(test_loss_d[-6:-1])))
        # ax.set_xlabel('Epoch', fontsize=xylabelsize)
        ax.set_ylabel(ylabel3, fontsize=xylabelsize)
        
        # if yrange is not None:
        #     ax.set_ylim(yrange)
        # else:
        if (ax.get_ylim()[1] - ax.get_ylim()[0]) > 1000:
            ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[0]+1000 ])
        if xrange is not None:
            ax.set_xlim(xrange)
        
        divider = make_axes_locatable(ax)
        resi_ax = divider.append_axes("bottom",size="20%", pad="3%")
        resi = test_loss_d - loss_d
        resi_ax.plot(epoch, resi, color=l1[0].get_color(), label='%.3g'%(np.mean(resi[-6:-1]))  )
        resi_ax.axhline(y=0, ls='--', color='k')
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
        ax.legend(loc='upper left',fontsize=legsize)
        resi_ax.legend(loc='lower right',fontsize=legsize)
    
    
    # coommon tick setup for all axsis
    for ax in axis.ravel():
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',top='on',right='on',direction='in' )
        ax.tick_params(axis='both',which='major', labelsize=ticklabelsize )
        ax.tick_params(axis='x',labelbottom=False)
        if grid:
            ax.grid(alpha=0.5)
        
    if c is not None:
        txt, title = make_txt_info(c, title=title, title_append=title_append)
        axis[0,0].text(0.96,0.96, '\n'.join(txt), ha='right', va='top', transform=axis[0,0].transAxes, fontsize=txtsize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))

        txt1, txt2 = make_txt_info_DA(c)
        axis[0,1].text(0.96,0.96, '\n'.join(txt1), ha='right', va='top', transform=axis[0,1].transAxes, fontsize=txtsize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))
        axis[0,2].text(0.96,0.96, '\n'.join(txt2), ha='right', va='top', transform=axis[0,2].transAxes, fontsize=txtsize,
                            bbox=dict(boxstyle='round', facecolor='w', alpha=0.8, edgecolor='silver'))

    
    
    fig.suptitle(title, fontsize=titlesize)
    fig.tight_layout()
    if figname:
        fig.savefig(figname,dpi=250)
        plt.close()
    # else:
    #     plt.show()
    return fig, axis
