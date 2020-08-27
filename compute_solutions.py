import numpy as np
import myAlgorithms as algs
import matplotlib.pyplot as plt
import os
import odl
import myFunctionals as fctls
import myOperators as ops


def deform_param(x, param):
    embedding = ops.Embedding_Affine(
                param.space, x.space.tangent_bundle)
    
    deform_op = odl.deform.LinDeformFixedDisp(embedding(param))
          
    return deform_op(x)
   
def deform(x):  
    return deform_param(x[0], x[1])

def save_image(x, filename):
    plt.imsave(filename, x, cmap='inferno')   

#%%
show = True

names = ['software_phantom_sinfo0', 'HV-109_sinfo1', 'HV-109_sinfoA', 'HV-114_sinfo9', 'HV-117_sinfo12', 'HV-118_sinfo0', 'HV-118_sinfoA']
names = ['HV-109_sinfoA', 'HV-118_sinfoA']
names = ['HV-109_sinfo1', 'HV-109_sinfoA', 'HV-114_sinfo9', 'HV-117_sinfo12', 'HV-118_sinfo0', 'HV-118_sinfoA']

names = []
#for i in [9, 14, 17, 18]:
for i in [9, 17]:
    for j in list(range(0, 15, 7)) + ['A']:
        names.append('HV-1{:02d}_sinfo{}'.format(i, j))
names.append('phantom_sinfo0')
names.pop(0)
names.pop(0)
names.pop(0)
names.pop(0)
names.pop(0)
names.pop(0)
names.pop(0)

names = ['HV-109_Lactate_sinfoA', 'HV-109_Hydrate_sinfoA', 'HV-109_Pyruvate_sinfoA']
#names = ['HV-109_Lactate_sinfoA', 'HV-109_Pyruvate_sinfoA']

stds = [0, 0.05, 0.1]
niters = [20, 30, 50, 100, 300, 500]
niters = [25, 50, 100, 100]

alphas = [1e-3, 5e-3, 1e-2]
alphas = [1e-3, 1e-2]
alphas = [5e-3, 1e-2, 5e-2]

gamma = 0.9995
strong_cvx = 1e-2
etas = [2e-2, 1e-2, 9e-3]
etas = [1e-2, 2e-2]
niter_prox = 20

init = 'zero'
#init = 'data'
#init = 'manual'


#%%
for name in names:
    folder_out = '../pics/{}'.format(name)
    
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
        
    dic = np.load('../processed_data/{}.npy'.format(name))
    data = dic['data']
    sinfo = dic['sinfo']
    
    U = odl.uniform_discr([-1, -1], [1, 1], sinfo.shape, interp='linear')
    V = odl.uniform_discr([-1, -1], [1, 1], data.shape, interp='linear')
    S = ops.Subsampling(U, V)
    
    #S = operators.get_sampling(U, shighres // sdata)
    np.random.seed(1807)
    #data = S.range.element(data_raw_small.ravel())
    data = S.range.element(data)
    Yaff = odl.tensor_space(6)
    
    # space of optimised variables
    X = odl.ProductSpace(U, Yaff)
    
    # Set some parameters and the general TV prox options
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = niter_prox

    reg_affine = odl.solvers.ZeroFunctional(Yaff)
    

    
    if init == 'zero':
        x0 = X.zero()
        
    elif init == 'data':
        x0 = X.element([S.adjoint(data), Yaff.zero()])
        
    elif init == 'manual':
            initname = 'std3.00e-02_izero_a1.00e-02_g1.00e+00_e5.00e-02_nprox20_scvx1.00e+00_iter50.npy'
            initname = 'std3.00e-02_imanual_a1.00e-03_g1.00e+00_e5.00e-02_nprox10_scvx1.00e+01_iter50.npy'
            initname = 'std3.00e-02_imanual_a1.00e-03_g1.00e+00_e1.00e-02_nprox10_scvx5.00e+01_iter100.npy'
            dic = np.load('../pics/HV-114/{}'.format(initname))
            x0 = X.element([dic['x'], dic['param']])
    
    else:
        x0 = None
        
    for alpha in alphas: # regularization parameter  
        for eta in etas:
            for std in stds:
                filename = '{}/std{:3.2e}_i{}_e{:3.2e}_a{:3.2e}_g{:5.4f}_nprox{}_scvx{:3.2e}'.format(folder_out, std, init, eta, alpha, gamma, niter_prox, strong_cvx)
                suptitle_param = 'std{:3.2e}, e{:3.2e}, a{:3.2e}'.format(std, eta, alpha)
                
                
                if std > 0:
                    K = odl.uniform_discr([-.1, -.1], [.1, .1], (25, 25), interp='linear')
                    k = K.element(lambda x: np.exp(-(x[0]**2 + x[1] ** 2)/std**2))
                    k /= k.inner(K.one())
                    C = ops.Convolution(U, k)
                else:
                    C = odl.IdentityOperator(U)
                    
                # Define smooth part (data term)
                f = fctls.DataFitL2Disp(X, data, S * C)  
                
                def show_result(x, param=None):
                    
                    if param is None:
                        param = x[1]
                        x = x[0]
                        
                    x_deformed_param = deform_param(x, param)
                    
                    plt.figure(1)
                    plt.clf()
                    
                    plt.subplot(231)
                    plt.imshow(x0[0], cmap='inferno')
                    plt.colorbar()
                    plt.title('init')
                    
                    plt.subplot(232)
                    plt.imshow(x, cmap='inferno')
                    plt.colorbar()
                    plt.title('result')
                    
                    plt.subplot(233)
                    plt.imshow(sinfo, cmap='gray')
                    plt.colorbar()
                    plt.title('sinfo')
                    
                    plt.subplot(234)
                    plt.imshow((S * C)(x_deformed_param), cmap='inferno')
                    plt.colorbar()
                    plt.title('estimated data')
                    
                    plt.subplot(235)
                    plt.imshow(data, cmap='inferno')
                    plt.colorbar()
                    plt.title('data')
                
                    plt.subplot(236)
                    e = (S * C)(x_deformed_param) - data
                    s = np.maximum(np.abs(np.max(e)), np.abs(np.min(e)))
                    plt.imshow(e, vmin=-s, vmax=s, cmap='RdBu')
                    plt.colorbar()
                    plt.title('est - data')
        
                reg_im = fctls.directionalTotalVariationNonnegative(U, alpha=alpha, sinfo=sinfo, 
                         gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx, prox_options=prox_options)
                
                g = odl.solvers.SeparableSum(reg_im, reg_affine) 
                    
                        
                cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                  odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                  odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True))
            
                
                #cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                #      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                #      odl.solvers.CallbackPrintTiming(fmt='total={:.3f} s', cumulative=True) &
                #      odl.solvers.CallbackPrint(func=obj, fmt='obj={:3.2e}'))
    
                L = [1e+2, 1e+16]
                L = [1, 1e+2]
                
                ud_vars = [0, 1]
                
                #%%
                palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L, tol=1e-10)
                
                #%%
                fig = plt.figure(2)
                plt.clf()
                for niter in niters:
                    palm.run(niter)
                    
                    show_result(palm.x)
                    plt.gcf().suptitle('{} {} {}'.format(name, suptitle_param, palm.niter))
                    plt.savefig('{}_iter{}.png'.format(filename, palm.niter))
        
                    palm.x[1].show(fig=fig)
                
                plt.savefig('{}_iter{}_deformation.png'.format(filename, palm.niter))
            
                save_image(palm.x[0], '{}_recon.png'.format(filename))
                
                with open('{}_iter{}.npy'.format(filename, palm.niter), 'wb') as file_out:
                    np.savez(file_out, x=palm.x[0], param=palm.x[1])