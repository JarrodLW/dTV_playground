import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktransform
import pydicom
import os
import myAlgorithms as algs

import nibabel as nib
import scipy.io as sio

import odl
import myOperators as ops

import myFunctionals as fctls

#from myModel import show_results
#import scipy.io as sio

folder_out = '../processed_data'


def save_sinfo(x, filename):
    plt.imsave(filename, x, cmap='gray')   

def save_data(x, filename):
    plt.imsave(filename, x, cmap='inferno')   
    
if not os.path.exists(folder_out):
    os.makedirs(folder_out)

    
def deform_param(x, param):
    embedding = ops.Embedding_Affine(
                param.space, x.space.tangent_bundle)
    
    deform_op = odl.deform.LinDeformFixedDisp(embedding(param))
          
    return deform_op(x)
   
def deform(x):  
    return deform_param(x[0], x[1])

def plot_and_save_data(data, sinfo, file_out, file_data, file_sinfo, amplify_data=1):
    
    sinfo = sinfo.copy()
    
    sdata = (64, 64)
    if data.shape != sdata:
        data = sktransform.resize(data, sdata)
    
    #U = odl.uniform_discr([-1, -1], [1, 1], data.shape)
    V = odl.uniform_discr([-1, -1], [1, 1], data.shape, interp='linear')
    data = V.element(data)
    
    data /= np.max(data)
    data = np.log10(10*data+1)
    data /= np.max(data)
    
    
    #xx = np.sort(data.asarray().ravel())
    #data /= xx[int(0.99 * len(xx))]
    
    #sinfo /= np.max(sinfo)
    #sinfo = np.log10(1*sinfo+1)
    sinfo /= np.max(sinfo)
    
        # Set some parameters and the general TV prox options
    prox_options = {}
    prox_options['name'] = 'FGP'
    prox_options['warmstart'] = True
    prox_options['p'] = None
    prox_options['tol'] = None
    prox_options['niter'] = 200
    g = fctls.TotalVariationBoxContraint(sinfo.space, alpha=1e-5, lower=0, 
                        prox_options=prox_options)
    sinfo = g.proximal(1)(sinfo)
    
    sinfo = estimate_deformation(data, sinfo)

    plt.clf()
    plt.subplot(221)
    plt.imshow(data, cmap='inferno', vmax=1)
    plt.title('{}'.format(file_data[:10]))
    plt.colorbar()
    
    plt.subplot(223)
    plt.imshow(sinfo, cmap='gray')
    plt.title('{}'.format(file_sinfo[:10]))
    plt.colorbar()
    
    plt.subplot(122)
    x = np.zeros((data.shape[0], data.shape[1], 3))
    x[:,:,0] = data * 2.1 * amplify_data
    x[:,:,1] = sktransform.resize(sinfo, data.shape)
    plt.imshow(x)
    plt.title('r: data, g: sinfo')
    
    plt.show()

    plt.savefig('{}.png'.format(file_out))
    
    with open('{}.npy'.format(file_out), 'wb') as f:
        np.savez(f, data=data, sinfo=sinfo)    
        
    save_sinfo(sinfo, '{}_sinfo.png'.format(file_out))
    save_data(data, '{}_data.png'.format(file_out))



def estimate_deformation(data, sinfo, filename=None):
    
    niter = 100
    niter = 250
    alpha = 5e-2
    gamma = 1
    strong_cvx = 1e-2
    eta = 1e-2
    eta = 5e-3
    
    U = sinfo.space
    V = data.space
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
    prox_options['niter'] = 20
    reg_affine = odl.solvers.ZeroFunctional(Yaff)
    
    C = odl.IdentityOperator(U)
    # Define smooth part (data term)
    f = fctls.DataFitL2Disp(X, data, S * C)  
    
    
    x0 = X.zero()
    #x0[0] = S.adjoint(data)
    
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
    
    L = [1, 1e+2]
    ud_vars = [0, 1]
        
    palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), niter=niter, callback=cb, L=L, tol=1e-8)
    
    print(palm.x[1])
    
    
    if filename is not None:
        show_result(palm.x)
        plt.savefig('{}.png'.format(filename))

    return deform_param(sinfo, palm.x[1])        
            

#%%
name = 'phantom'

folder_data = '../data/h13c_18_2_14_h13c_18_2_14_5961'
#file_data = ['0015', '0063', '0111', '0159', '0207']
file_data = '011_MNSRP_TS48_MET5_SL1_LPAPB_ideal/0159'
file_sinfo = '012_scout_confirm_position/0003'


data_dcm = pydicom.read_file('{}/{}.dcm'.format(folder_data, file_data))    
data_raw = data_dcm.pixel_array.astype('float64')
sinfo_dcm = pydicom.read_file('{}/{}.dcm'.format(folder_data, file_sinfo))
sinfo_raw = sinfo_dcm.pixel_array.astype('float64')

#data_raw /= np.max(data_raw)
#sinfo_raw /= np.max(sinfo_raw)
    
U = odl.uniform_discr([-1, -1], [1, 1], sinfo_raw.shape, interp='linear')
V = odl.uniform_discr([-1, -1], [1, 1], data_raw.shape, interp='linear')
data = V.element(data_raw)
Yaff = odl.tensor_space(6)

scale = 0.65
phi = 0.01
shift = [0,-0.06]
param = [shift[0], shift[1], scale * np.cos(phi) - 1, scale * np.sin(phi), scale * -np.sin(phi), scale * np.cos(phi) - 1]
sinfo = deform_param(U.element(sinfo_raw), Yaff.element(param))

file_data = name
file_out = '{}/{}_sinfo{}'.format(folder_out, file_data, 0)
plot_and_save_data(data_raw, sinfo, file_out, file_data, file_sinfo, amplify_data=1)


#%%
name = 'HV-109'
file_data = name
folder_data = '../data/{}'.format(name)
file_sinfo = 'HV-109_20809_2_Stealth_(3D_Bravo)_20171128.img'

#mat = sio.loadmat('{}/{}new.mat'.format(folder_data, name))
## last index one takes the middle slice, pyruvate?
#data_raw = np.absolute(mat['bb_sum'][:,:,3, 1]).astype('float64')
##data_raw = np.absolute(mat['bb'][:,:,5, 3, 1]).astype('float64')
##data_raw /= np.max(data_raw)
##data_raw *= 3

s_molecule = ['Lactate', 'Hydrate', 'Alanine', 'Pyruvate', 'Bicarbonate']

#Lactate (1:48), the main metabolic product of interest
#Hydrate(49:96), a shorter-lived byproduct
#Alanine(97:144), blank in this dataset
#Pyruvate(145:192), the precursor molecule
#Bicarbonate(193:240), blank in this dataset

molecules = [0, 1, 3]


mat = sio.loadmat('{}/{}-matrix-recons.mat'.format(folder_data, name))

for molecule in molecules:

    data_raw = np.absolute(np.sum(mat['bb64'][:,:,:,molecule, 1], 2)).astype('float64')
    
    
    img = nib.load('{}/{}'.format(folder_data, file_sinfo))
    sinfo_raw = img.get_fdata()
    slices = range(84-7, 84+8)
    #slices = range(84-7, 84-6)
    sinfos = [np.rot90(sinfo_raw[:, :, i, 0], 1) for i in slices]
    
        
    U = odl.uniform_discr([-1, -1], [1, 1], sinfos[0].shape, interp='linear')
    V = odl.uniform_discr([-1, -1], [1, 1], data_raw.shape, interp='linear')
    data = V.element(data_raw)
    Yaff = odl.tensor_space(6)
    
    scale = 1
    phi = .1
    shift = [0.15,.02]
    param = [shift[0], shift[1], scale * np.cos(phi) - 1, scale * np.sin(phi), scale * -np.sin(phi), scale * np.cos(phi) - 1]
    sinfos = [deform_param(U.element(sinfo), Yaff.element(param)) for sinfo in sinfos]
    
    #file_out = '{}/{}_sinfo{}'.format(folder_out, file_data, 0)
    #plot_and_save_data(data_raw, sinfos[0], file_out, file_data, file_sinfo, amplify_data=1)
       
    for i, sinfo in enumerate(sinfos):
        sinfo[:,205:] = 0
        file_out = '{}/{}_{}_sinfo{}'.format(folder_out, file_data, s_molecule[molecule], i)
        plot_and_save_data(data_raw, sinfo, file_out, file_data, file_sinfo, amplify_data=1)
    
    file_out = '{}/{}_{}_sinfo{}'.format(folder_out, file_data, s_molecule[molecule], 'A')
    plot_and_save_data(data_raw, sum(sinfos), file_out, file_data, file_sinfo, amplify_data=1)

#%%
names = ['HV-114', 'HV-117', 'HV-118']
#bames = ['HV-117']
#names = ['HV-118']

for name in names:
    folder_sinfo = '002_Stealth_3D_Bravo_'
    file_data = name
    
    folder_data = '../data/{}'.format(name)
        
    
    mat = sio.loadmat('{}/{}new.mat'.format(folder_data, name))
    # last index one takes the middle slice, pyruvate?
    data_raw = np.absolute(mat['bb_sum'][:,:,3, 1]).astype('float64')
    #data_raw = np.absolute(mat['bb'][:,:,5, 3, 1]).astype('float64')
    #data_raw /= np.max(data_raw)
    #data_raw *= 3
  
    
    if name[-1] == '4':
        slices = range(64-7, 64+8)
        scale = 0.9
        phi = 0
        shift = [0,0]
        
    elif  name[-1] == '7':
        slices = range(50-7, 50+8)
        scale = 0.9
        phi = 0
        shift = [0.1,0]
        
    elif name[-1] == '8':
        slices = range(51-7, 51+8)
        scale = 0.9
        phi = 0
        shift = [0.1,0]
       
    else:
        scale = None
    
    
    sinfos = []
    file_sinfos = ['{:04d}'.format(i) for i in slices]
    
    for file_sinfo in file_sinfos:
        sinfo_dcm = pydicom.read_file('{}/{}/{}.dcm'.format(folder_data, folder_sinfo, file_sinfo))    
    #    print(fsinfo,sinfo_dcm.PixelSpacing, sinfo_dcm.Columns, sinfo_dcm.Rows, sinfo_dcm.ImagePositionPatient)
        sinfos.append(sinfo_dcm.pixel_array.astype('float64'))
        
    U = odl.uniform_discr([-1, -1], [1, 1], sinfos[0].shape, interp='linear')
    V = odl.uniform_discr([-1, -1], [1, 1], data_raw.shape, interp='linear')
    data = V.element(data_raw)
    Yaff = odl.tensor_space(6)
        
    param = [shift[0], shift[1], scale * np.cos(phi) - 1, scale * np.sin(phi), scale * -np.sin(phi), scale * np.cos(phi) - 1]
    sinfos = [deform_param(U.element(sinfo), Yaff.element(param)) for sinfo in sinfos]
    
    #file_out = '{}/{}_sinfo{}'.format(folder_out, file_data, 7)
    #plot_and_save_data(data_raw, sinfos[0], file_out, file_data, file_sinfo, amplify_data=1)
     
    #sinfos = [sinfos[0]]
    
    #%
    file_sinfo = folder_sinfo
    for i, sinfo in enumerate(sinfos):
        sinfo[:, :40] = 0
        file_out = '{}/{}_sinfo{}'.format(folder_out, file_data, i)
        plot_and_save_data(data_raw, sinfo, file_out, file_data, file_sinfo, amplify_data=1)
    
    file_out = '{}/{}_sinfo{}'.format(folder_out, file_data, 'A')
    plot_and_save_data(data_raw, sum(sinfos), file_out, file_data, file_sinfo, amplify_data=1)
