import astra
import h5py
import numpy as np
from scipy.ndimage import interpolation

filename = 'Experiment1_XRF.hdf5'
f1 = h5py.File(filename, 'r+')

sino_Co = np.array(f1['sino_Co'])
sino_Co_1 = sino_Co[:, :, 0]

def recon_astra(sinogram, center, angles=None, method="FBP", num_iter=1, win="hann", pad=0): #, ratio=1.0):
    # Taken from Vo's code
    """
    Wrapper of reconstruction methods implemented in the astra toolbox package.
    https://www.astra-toolbox.com/docs/algs/index.html
    ---------
    Parameters: - sinogram: 2D tomographic data.
                - center: center of rotation.
                - angles: tomographic angles in radian.
                - ratio: apply a circle mask to the reconstructed image.
                - method: Reconstruction algorithms
                    for CPU: 'FBP', 'SIRT', 'SART', 'ART', 'CGLS'.
                    for GPU: 'FBP_CUDA', 'SIRT_CUDA', 'SART_CUDA', 'CGLS_CUDA'.
                - num_iter: Number of iterations if using iteration methods.
                - filter: apply filter if using FBP method:
                    'hamming', 'hann', 'lanczos', 'kaiser', 'parzen',...
                - pad: padding to reduce the side effect of FFT.
    ---------
    Return:     - square array.
    """
    if pad > 0:
        sinogram = np.pad(sinogram, ((0, 0), (pad, pad)), mode='edge')
        center = center + pad
    (nrow, ncol) = sinogram.shape
    if angles is None:
        angles = np.linspace(0.0, 180.0, nrow) * np.pi / 180.0
    proj_geom = astra.create_proj_geom('parallel', 1, ncol, angles)
    vol_geom = astra.create_vol_geom(ncol, ncol)
    cen_col = (ncol - 1.0) / 2.0
    shift = cen_col - center
    sinogram = interpolation.shift(sinogram, (0, shift), mode='nearest')
    sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict(method)
    proj_id = astra.creators.create_projector('line', proj_geom, vol_geom) # new code
    cfg['ProjectionDataId'] = sino_id
    cfg['ProjectorId'] = proj_id # new code
    cfg['ReconstructionDataId'] = rec_id
    if method == "FBP_CUDA":
        cfg["FilterType"] = win
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, num_iter)
    rec = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    if pad > 0:
        rec = rec[pad:-pad, pad:-pad]
    #if not (ratio is None):
    #    rec = rec * circle_mask(rec.shape[0], ratio)
    return rec

# computing the reconstruction using FBP

center = 83
angle_array = np.arange(180) * np.pi/180

recon = recon_astra(sino_Co_1, center, angles=angle_array)

