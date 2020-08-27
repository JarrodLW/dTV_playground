import numpy as np

#import sys

#sys.path.append('/Users/jlw31/PycharmProjects/dTV_playground/code/')

import myAlgorithms as algs
import matplotlib.pyplot as plt
import os
import odl
import myFunctionals as fctls
import myOperators as ops
from Utils import *
import math

def deform_param(x, param):
    embedding = ops.Embedding_Affine(
        param.space, x.space.tangent_bundle)

    deform_op = odl.deform.LinDeformFixedDisp(embedding(param))

    return deform_op(x)

alpha = 100000
eta = 0.01
gamma = 0.995
strong_cvx = 1e-2
niter_prox = 20
niter = 50

Yaff = odl.tensor_space(6)

data_vec = np.fromfile("fid", dtype=np.int32)
data = recasting_fourier_as_complex(data_vec, 128, 256)
height, width = data.shape
data_fft = np.fft.fftshift(data)

sinfo_before_tweaking = np.fft.fftshift(np.fft.ifft2(data_fft))

complex_space = odl.uniform_discr(min_pt=[-height//2, -width//2], max_pt=[height//2, width//2],
                                      shape=[height, width], dtype='complex', interp='linear')
image_space = complex_space.real_space ** 2
forward_op = ops.RealFourierTransform(image_space)

data = np.array([np.real(data_fft), np.imag(data_fft)])

# downsampling = True
# downsampling_factor = 2
#
# if downsampling:
#
#     data_space = odl.ProductSpace(odl.uniform_discr([ 0.,  0.], [127.,  127.], (height//downsampling_factor,
#                                                                                 width//downsampling_factor), nodes_on_bdry=True), 2)
#     S = ops.Subsampling(forward_op.range, data_space)
#     data_odl = S(forward_op.range.element(data))
#     forward_op = S * forward_op
#
# else:

#sample_rate = 0.4
#subsampling_arr, _ = horiz_rand_walk_mask(height, width, round(sample_rate*height), allowing_inter=False, p=[0.1, .8, 0.1])
subsampling_arr = np.zeros((height, width))
subsampling_arr[6*height//16 : 10*height//16, 6*width//16: 10*width//16] = 1
subsampling_arr = np.fft.fftshift(subsampling_arr)
#subsampling_arr = bernoulli_mask(height, width, expected_sparsity=sample_rate)[0]
subsampling_arr_doubled = np.array([subsampling_arr, subsampling_arr])

data += 10 * odl.phantom.white_noise(forward_op.range).asarray()

data = subsampling_arr*data
forward_op = forward_op.range.element(subsampling_arr_doubled) * forward_op

#J = ops.Complex2Real(complex_space)
#data_odl = J(data_complex_odl)
data_odl = forward_op.range.element(data)


#sinfo = image_space.element(J(sinfo))
sinfo = complex_space.real_space.element(np.abs(sinfo_before_tweaking))
sinfo = sinfo*(sinfo - np.max(sinfo))/40
sinfo = sinfo - np.min(sinfo)


#params = Yaff.element([2, 2, np.cos(math.pi/90) - 1, -np.sin(math.pi/90), np.sin(math.pi/90), np.cos(math.pi/90) - 1])
#sinfo = deform_param(sinfo, params)

# space of optimised variables
X = odl.ProductSpace(image_space, Yaff)

# Set some parameters and the general TV prox options
prox_options = {}
prox_options['name'] = 'FGP'
prox_options['warmstart'] = True
prox_options['p'] = None
prox_options['tol'] = None
prox_options['niter'] = niter_prox

reg_affine = odl.solvers.ZeroFunctional(Yaff)
x0 = X.zero()

f = fctls.DataFitL2Disp(X, data_odl, forward_op)

reg_im = fctls.directionalTotalVariationNonnegative(image_space, alpha=alpha, sinfo=sinfo,
                                                    gamma=gamma, eta=eta, NonNeg=True, strong_convexity=strong_cvx,
                                                    prox_options=prox_options)

g = odl.solvers.SeparableSum(reg_im, reg_affine)

cb = (odl.solvers.CallbackPrintIteration(end=', ') &
      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
      odl.solvers.CallbackPrintTiming(fmt='total={:.3f}s', cumulative=True))

L = [1, 1e+2]
ud_vars = [0]

# %%
palm = algs.PALM(f, g, ud_vars=ud_vars, x=x0.copy(), callback=cb, L=L)
palm.run(niter)

recon = palm.x[0].asarray()
recon_complex = recon[0] + 1j*recon[1]

plt.figure()
plt.imshow(np.abs(recon_complex), cmap=plt.cm.gray)
plt.axis('off')

inv = ops.RealFourierTransform(image_space).inverse(data_odl)
inv_complex = inv.asarray()[0] + 1j * inv.asarray()[1]
plt.figure()
plt.imshow(np.abs(inv_complex), cmap=plt.cm.gray)
plt.axis('off')

plt.figure()
plt.imshow(sinfo, cmap=plt.cm.gray)
plt.axis('off')
plt.colorbar()

plt.figure()
plt.imshow(np.abs(sinfo_before_tweaking), cmap=plt.cm.gray)
plt.axis('off')
plt.colorbar()

#sinfo_shifted = sinfo - np.amin(sinfo)

plt.figure()
plt.hist(np.abs(sinfo_before_tweaking).ravel(), bins=100, label='Groundtruth', alpha=0.5)
plt.legend()
plt.hist(sinfo.asarray().ravel(), bins=100, label='Guide image', alpha=0.5)
#plt.hist(sinfo_shifted.asarray().ravel(), bins =100, range = (0, 25000), label='Guide image')
plt.legend()
plt.xlabel("Pixel intensity")
plt.ylabel("Counts")

#
# plt.figure()
# plt.imshow(np.log(np.abs(data_fft)), cmap=plt.cm.gray)
#
#
# plt.figure()
# plt.imshow(np.log(np.abs(np.fft.fftshift(data_fft))), cmap=plt.cm.gray)

plt.figure()
plt.imshow(np.log(1+np.abs(np.fft.fftshift(data[0] + 1j*data[1]))), cmap=plt.cm.gray)
plt.axis("off")

np.count_nonzero(subsampling_arr)/128**2

plt.show()


