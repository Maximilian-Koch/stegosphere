import numpy as np


from stegosphere import utils


def mse(array_1, array_2):
    """
    Calculates the Mean Squared Error (MSE) between the original and encoded array.

    :return: MSE value
    :rtype: float
    """
    return np.mean((array_1.astype(np.float64) - array_2.astype(np.float64)) ** 2)


def psnr(array_1, array_2, max_i=None):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between the original and encoded array.
    
    :return: PSNR value in dB.
    :rtype: float
    """
    mse_val = mse(array_1, array_2)
    if mse_val == 0:
        return float('inf')
    if max_i is None:
        max_i = utils.dtype_range(array_1.dtype)[1]
    return 10 * np.log10((max_i ** 2) / mse_val)
    

def ssim(array_1, array_2):
    from skimage.metrics import structural_similarity as ss
    return ss(array_1, array_2)


def manhattan_distance(array_1, array_2):
    return np.sum(np.abs(array_1-array_2))

def euclidean_distance(array_1, array_2):
    return np.sqrt(np.sum((array_1-array_2)**2))

def chi_distance(array_1, array_2, epsilon=1e-10):
    p = array_1 / (array_1.sum() + epsilon)
    q = array_2 / (array_2.sum() + epsilon)


def kl_divergence(array_1, array_2, epsilon=1e-10):
    """
    Kullback-Leibler divergence
    """
    p,q = array_1, array_2

    #normalize
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p /= (p.sum() + epsilon)
    q /= (q.sum() + epsilon)

    kl = np.sum(p * np.log((p + epsilon) / (q + epsilon)))
    return kl


def js_divergence(array_1, array_2):
    """
    Jensen-Shannon divergence
    """
    p,q = array_1, array_2

    #normalize
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p /= (p.sum() + epsilon)
    q /= (q.sum() + epsilon)
    
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
    kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))
    js = 0.5 * (kl_pm + kl_qm)
    return js

def bhattacharyya_divergence(array_1, array_2):
    p,q = array_1, array_2

    #normalize
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p /= (p.sum() + epsilon)
    q /= (q.sum() + epsilon)
    bc = np.sum(np.sqrt(p*q))
    ba = -np.log(bc+epsilon)
    return ba

def results(array_1, array_2, measures='all', epsilon=1e-10, psnr_max_i=None):
    if measures == 'all' :
        measures = ['mse','psnr','ssim','kl','js','bhattacharyya','manhattan','euclidean','chi']
    #mse, psnr, ssim, kl, js, bhattacharyya, manhattan, euclidean, chi

    output = {}
    if 'psnr' in measures or 'mse' in measures:
        mse = np.mean((self.before.astype(np.float64) - self.after.astype(np.float64)) ** 2)
        if 'mse' in measures:
            output.update({'mse' : mse})
    if 'psnr' in measures:
        if mse == 0:
            psnr = float('inf')
        else:
            if psnr_max_i is None:
                psnr_max_i = utils.dtype_range(self.before.dtype)[1]
            psnr = 10 * np.log10((psnr_max_i ** 2) / mse)
        output.update({'psnr' : psnr})
        
    if 'ssim' in measures:
        pass
    if 'kl' in measures or 'js' in measures or 'bhattacharyya' in measures:
        p,q = array_1, array_2

        #normalize
        p = p.astype(np.float64)
        q = q.astype(np.float64)
        p /= (p.sum() + epsilon)
        q /= (q.sum() + epsilon)
        if 'kl' in measures:
        #Kullback-Leibler divergence
            kl = np.sum(p * np.log((p + epsilon) / (q + epsilon)))
            output.update({'kl':kl})
        if 'js' in measures:
            #Jensen-Shannon divergence
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log((p + epsilon) / (m + epsilon)))
            kl_qm = np.sum(q * np.log((q + epsilon) / (m + epsilon)))
            js = 0.5 * (kl_pm + kl_qm)
            output.update({'js':js})
        if 'bhattacharyya' in measures:
            bc = np.sum(np.sqrt(p*q))
            ba = -np.log(bc+epsilon)
            output.update({'bhattacharyya' : ba})
    if 'manhattan' in measures or 'euclidean' in measures or 'chi' in measures:
        diff = array_1 - array_2
        if 'euclidean' or 'chi' in measures:
            diffsquare = diff**2
            if 'euclidean' in measures:
                euclidean = np.sqrt(np.sum(diffsquare))
                output.update({'euclidean' : euclidean})
            if 'chi' in measures:
                chi = np.sum(diffsquare / (array_1+array_2+epsilon))
                output.update({'chi' : chi})
        if 'manhattan' in measures:
            manhattan = np.sum(np.abs(diff))
            output.update({'manhattan' : manhattan})
            
    return output
