import numpy as np
import torch


def layernorm_forward(x, gamma, beta, eps):
    out, cache = None, None
    M, _ = x.shape
    x_mean = np.mean(x, axis=1).reshape(M, 1)
    x_var = np.var(x, axis=1).reshape(M, 1)

    xmu = x - x_mean
    xivar = np.sqrt(x_var + eps)
    xhat = xmu / xivar

    out = gamma * xhat + beta
    cache = (x, xmu, xivar, x_mean, xhat, gamma)
    return out, cache



def pytorch_layernorm_backward(x, gamma, beta, eps, dy):
    _, K = x.shape

    layer_norm = torch.nn.LayerNorm(K, eps=eps)
    layer_norm.weight.data = torch.from_numpy(gamma)
    layer_norm.bias.data = torch.from_numpy(beta)

    tx = torch.from_numpy(x)
    tx.requires_grad = True
    ty = layer_norm.forward(tx)
    tdy = torch.from_numpy(dy)
    ty.backward(tdy)

    return tx.grad.numpy(), layer_norm.weight.grad.numpy(), layer_norm.bias.grad.numpy()


def layernorm_nogamma_beta_backward(dy, cache):
    # https://github.com/pytorch/pytorch/blob/9af0e476539041db99ca0cc05a5a11de209fdf38/caffe2/python/operator_test/layer_norm_op_test.py#L43
    dx = None

    (x, xmu, xivar, x_mean, xhat, _) = cache
    M, K = x.shape

    dstdev_end = (-1.0) / np.power(xivar, 2.0) * \
        np.sum(xmu * dy, axis=1).reshape([M, 1])

    dmean_end = np.sum(-1.0 / xivar * dy, axis=1).reshape([M, 1])
    dx_end = 1.0 / xivar * dy

    # stdev block
    dmean_stdev = -1.0 * x_mean / xivar * dstdev_end
    dx_stdev = x / (K * xivar) * dstdev_end

    # mean block
    dmean = dmean_end + dmean_stdev
    dxmean = (1.0 / K) * dmean

    # final outputs
    dx = dx_end + dx_stdev + dxmean
    return dx


def rocking_layernorm_backward(dy, cache):
    dx, dgamma, dbeta = None, None, None

    (x, xmu, xivar, x_mean, xhat, gamma) = cache
    M, K = x.shape

    dgamma = np.sum(dy * xhat, axis=0, keepdims=True)
    dbeta = np.sum(dy, axis=0, keepdims=True)

    dlxhat = dy * gamma
    dxhatx = 1 / xivar

    dlvar = -0.5 * np.sum(dlxhat * xmu * xivar ** (-3), axis=1, keepdims=True)
    dvarx = 2 / K * (xmu - 1 / K * np.sum(xmu, axis=1, keepdims=True))

    dlmu = -np.sum(dlxhat / xivar, axis=1, keepdims=True)
    dmux = 1 / K

    dx = dlxhat * dxhatx + dlvar * dvarx + dlmu * dmux

    return dx, dgamma, dbeta


def c9_layernorm_backward(dy, cache):
    # https://zhuanlan.zhihu.com/p/38040024
    dx, dgamma, dbeta = None, None, None

    (x, xmu, xivar, x_mean, xhat, gamma) = cache
    _, K = x.shape

    dgamma = np.sum(dy * xhat, axis=0, keepdims=True)
    dbeta = np.sum(dy, axis=0, keepdims=True)

    dlxhat = dy * gamma
    dxhatx = 1 / xivar

    dlvar = -0.5 * np.sum(gamma * xmu * xivar ** (-3)
                          * dy, axis=1, keepdims=True)
    dvarx = 2 * xmu / K

    dlmu = -1. * np.sum(dlxhat / xivar, axis=1, keepdims=True) - \
        2. * np.sum(dlvar * xmu, axis=1, keepdims=True) / K
    dmux = 1 / K

    dx = dlxhat * dxhatx + dlvar * dvarx + dlmu * dmux
    return dx, dgamma, dbeta


def cs231n_layernorm_backward(dy, cache):
    # Imitate batchnorm backward
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    dx, dgamma, dbeta = None, None, None

    (x, xmu, xivar, x_mean, xhat, gamma) = cache
    M, K = x.shape

    # step9
    dbeta = np.sum(dy, axis=0, keepdims=True)
    dgammax = dy  # not necessary, but more understandable

    # step8
    dgamma = np.sum(dgammax*xhat, axis=0, keepdims=True)
    dxhat = dgammax * gamma

    # step7
    divar = np.sum(dxhat*xmu, axis=1, keepdims=True)
    dxmu1 = dxhat / xivar

    # step6
    dsqrtvar = -1. / (xivar**2) * divar

    # step5
    dvar = 0.5 * 1. / xivar * dsqrtvar

    # step4
    dsq = 1. / K * np.ones((M, K)) * dvar

    # step3
    dxmu2 = 2 * xmu * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=1, keepdims=True)

    # step1
    dx2 = 1. / K * np.ones((M, K)) * dmu

    # step0
    dx = dx1 + dx2

    return dx, dgamma, dbeta


def kevin_layernorm_backward(dy, cache):
    # Imitate batchnorm backward
    # https://kevinzakka.github.io/2016/09/14/batch_normalization/
    dx, dgamma, dbeta = None, None, None

    (x, _, xivar, _, xhat, gamma) = cache
    M, K = x.shape

    # intermediate partial derivatives
    dxhat = dy * gamma
    inv_var = 1./xivar

    # final partial derivatives
    dx = (1. / K) * inv_var * (K * dxhat - np.sum(dxhat, axis=1, keepdims=True)
                               - xhat * np.sum(dxhat * xhat, axis=1, keepdims=True))
    dbeta = np.sum(dy, axis=0, keepdims=True)
    dgamma = np.sum(xhat * dy, axis=0, keepdims=True)

    return dx, dgamma, dbeta


def moreh_layernorm_backward(dy, cache):
    dx, dgamma, dbeta = None, None, None

    (x, xmu, xivar, x_mean, xhat, gamma) = cache
    M, K = x.shape

    ds = np.sum(dy * gamma * x, axis=1, keepdims=True)
    db = np.sum(dy * gamma, axis=1, keepdims=True)

    a = (db * x_mean - ds) * xivar ** (-3) / K
    c2 = -(a * x_mean + db / xivar / K)
    dx = dy * gamma / xivar + a * x + c2

    dgamma = np.sum(dy * xhat, axis=0, keepdims=True)
    dbeta = np.sum(dy, axis=0, keepdims=True)

    return dx, dgamma, dbeta


if __name__ == '__main__':
    M = 2
    K = 4
    x = np.random.randn(M, K).astype(np.float32)
    gamma = np.random.randn(K).astype(np.float32)
    beta = np.random.randn(K).astype(np.float32)
    eps = 1e-5

    dy = np.random.randn(M, K).astype(np.float32)

    y, cache = layernorm_forward(x, gamma, beta, eps)
    dx, dgamma, dbeta = pytorch_layernorm_backward(x, gamma, beta, eps, dy)
    dx2, dgamma2, dbeta2 = rocking_layernorm_backward(dy, cache)
    dx3, dgamma3, dbeta3 = cs231n_layernorm_backward(dy, cache)
    dx4, dgamma4, dbeta4 = kevin_layernorm_backward(dy, cache)
    dx5, dgamma5, dbeta5 = c9_layernorm_backward(dy, cache)
    dx6 = moreh_layernorm_backward(dy, cache)

    print('--------pytorch dx--------')
    print(dx)
    print('--------dx2--------')
    print(dx2)
    print('--------dx3--------')
    print(dx3)
    print('--------dx4--------')
    print(dx4)
    print('--------dx5--------')
    print(dx5)
    print('--------dx6--------')
    print(dx6)
