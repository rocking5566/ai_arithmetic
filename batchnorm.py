import numpy as np


def batchnorm_forward(x, gamma, beta, eps):
    N, D = x.shape

    # step1: calculate mean
    mu = 1./N * np.sum(x, axis=0)

    # step2: subtract mean vector of every trainings example
    xmu = x - mu

    # step3: following the lower branch - calculation denominator
    sq = xmu ** 2

    # step4: calculate variance
    var = 1./N * np.sum(sq, axis=0)

    # step5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)

    # step6: invert sqrtwar
    ivar = 1./sqrtvar

    # step7: execute normalization
    xhat = xmu * ivar

    # step8: Nor the two transformation steps
    gammax = gamma * xhat

    # step9
    out = gammax + beta

    # store intermediate
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    return out, cache


def batchnorm_backward(dy, cache):
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    # unfold the variables stored in cache
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    # get the dimensions of the input/output
    N, D = dy.shape

    # step9
    dbeta = np.sum(dy, axis=0)
    dgammax = dy  # not necessary, but more understandable

    # step8
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma

    # step7
    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar

    # step6
    dsqrtvar = -1. / (sqrtvar**2) * divar

    # step5
    dvar = 0.5 * 1. / sqrtvar * dsqrtvar

    # step4
    dsq = 1. / N * np.ones((N, D)) * dvar

    # step3
    dxmu2 = 2 * xmu * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

    # step1
    dx2 = 1. / N * np.ones((N, D)) * dmu

    # step0
    dx = dx1 + dx2

    return dx, dgamma, dbeta


def kevin_batchnorm_backward(dy, cache):
    # https://kevinzakka.github.io/2016/09/14/batch_normalization/
    N, D = dy.shape
    x_hat, gamma, xmu, inv_var, sqrtvar, var, eps = cache

    # intermediate partial derivatives
    dxhat = dy * gamma

    # final partial derivatives
    dx = (1. / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0)
                               - x_hat*np.sum(dxhat*x_hat, axis=0))
    dbeta = np.sum(dy, axis=0)
    dgamma = np.sum(x_hat*dy, axis=0)

    return dx, dgamma, dbeta


def rocking_batchnorm_backward(dy, cache):
    N, D = dy.shape
    x_hat, gamma, xmu, inv_var, sqrtvar, var, eps = cache

    dx = (1./N * inv_var * gamma) * [N*dy - x_hat * np.sum(
        x_hat * dy, axis=0) + np.sum(1/N * x_hat * np.sum(x_hat * dy, axis=0) - dy, axis=0)]

    dbeta = np.sum(dy, axis=0)
    dgamma = np.sum(x_hat*dy, axis=0)

    return dx, dgamma, dbeta


def qianfeng_batchnorm_backward(dy, cache):
    N, D = dy.shape
    x_hat, gamma, xmu, inv_var, sqrtvar, var, eps = cache

    dbeta = np.sum(dy, axis=0)
    dgamma = np.sum(x_hat*dy, axis=0)
    dx = (1. / N) * inv_var * gamma * (N*dy - dbeta - x_hat*dgamma)

    return dx, dgamma, dbeta


if __name__ == '__main__':
    N = 2
    D = 4
    x = np.random.randn(N, D).astype(np.float32)
    gamma = np.random.randn(D).astype(np.float32)
    beta = np.random.randn(D).astype(np.float32)
    eps = 0.01

    dy = np.random.randn(N, D).astype(np.float32)

    y, cache = batchnorm_forward(x, gamma, beta, eps)
    dx, dgamma, dbeta = batchnorm_backward(dy, cache)
    dx2, dgamma2, dbeta2 = kevin_batchnorm_backward(dy, cache)
    dx3, dgamma3, dbeta3 = rocking_batchnorm_backward(dy, cache)
    dx4, dgamma4, dbeta4 = qianfeng_batchnorm_backward(dy, cache)

    print('--------dx--------')
    print(dx)
    print('--------dx2--------')
    print(dx2)
    print('--------dx3--------')
    print(dx3)
    print('--------dx4--------')
    print(dx4)
