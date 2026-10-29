"""Blocked energy and covariance statistics."""
import numpy as np

def energy_statistics(raw):
    n=len(raw);block_size=max(1,n//20);blocks=n//block_size
    usable=raw[:blocks*block_size].reshape(blocks,block_size,raw.shape[1])
    means=usable.mean(axis=1);mean=raw.mean(axis=0)
    se=means.std(axis=0,ddof=1)/np.sqrt(blocks) if blocks>1 else np.full(raw.shape[1],np.nan)
    covariance=np.cov(means[:,:2],rowvar=False,ddof=1)/blocks if blocks>1 else np.full((2,2),np.nan)
    # d(e_J,e_K)/dg=-2 Cov(Q_nn,e_J/e_K), including normalization derivative.
    grad=-2*(np.mean(raw[:,:2]*raw[:,4,None],axis=0)-mean[:2]*mean[4])
    block_grad=-2*((usable[:,:,:2]*usable[:,:,4,None]).mean(axis=1)-means[:,:2]*means[:,4,None])
    grad_se=block_grad.std(axis=0,ddof=1)/np.sqrt(blocks) if blocks>1 else np.full(2,np.nan)
    return dict(e_j=float(mean[0]),e_k=float(mean[1]),e_j_error=float(se[0]),e_k_error=float(se[1]),
                covariance_jk_of_mean=covariance.tolist(),imaginary_e_j=float(mean[2]),imaginary_e_k=float(mean[3]),
                imaginary_e_j_error=float(se[2]),imaginary_e_k_error=float(se[3]),
                q_nn=float(mean[4]),q_nn_error=float(se[4]),s_pi_pi=float(mean[5]),s_pi_pi_error=float(se[5]),
                m_cdw_squared=float(mean[5]),gradient_j_g=float(grad[0]),gradient_k_g=float(grad[1]),
                gradient_j_g_error=float(grad_se[0]),gradient_k_g_error=float(grad_se[1]),
                blocks=blocks,block_size=block_size)

