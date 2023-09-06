import numpy as np
import torch


def NHWGC2NCHW(x):
    N, H, W, G, C1 = x.shape
    C = G * C1
    x = x.reshape(N, H, W, C)
    return np.transpose(x, (0, 3, 1, 2))


def NCHW2NHWGC(x, G):
    C = x.shape[1]
    C1 = C // G
    return np.transpose(x, (0, 2, 3, 1)).reshape(N, H, W, G, C1)


def pytorch_groupnorm_forward(x, gamma, beta, eps):
    _, _, _, G, C1 = x.shape
    C = G * C1
    x = NHWGC2NCHW(x)
    tx = torch.from_numpy(x)
    gamma = gamma = gamma.reshape(C)
    beta = beta = beta.reshape(C)

    group_norm = torch.nn.GroupNorm(G, C, eps=eps)
    group_norm.weight.data = torch.from_numpy(gamma)
    group_norm.bias.data = torch.from_numpy(beta)
    ty = group_norm.forward(tx)

    y = ty.detach().numpy()
    return NCHW2NHWGC(y, G)


def pytorch_groupnorm_backward(x, gamma, beta, eps, dy):
    _, _, _, G, C1 = x.shape
    C = G * C1
    x = NHWGC2NCHW(x)
    tx = torch.from_numpy(x)

    dy = NHWGC2NCHW(dy)
    tdy = torch.from_numpy(dy)

    tx.requires_grad = True
    gamma = gamma = gamma.reshape(C)
    beta = beta = beta.reshape(C)

    group_norm = torch.nn.GroupNorm(G, C, eps=eps)
    group_norm.weight.data = torch.from_numpy(gamma)
    group_norm.bias.data = torch.from_numpy(beta)
    ty = group_norm.forward(tx)
    ty.backward(tdy)

    dx = tx.grad.numpy()
    dx = NCHW2NHWGC(dx, G)
    dgamma = group_norm.weight.grad.numpy().reshape(G, C1)
    dbeta = group_norm.bias.grad.numpy().reshape(G, C1)
    return dx, dgamma, dbeta


def groupnorm_forward(x, gamma, beta, eps):
    out, cache = None, None
    N, H, W, G, C = x.shape
    x_mean = np.mean(x, axis=(1, 2, 4), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 4), keepdims=True)

    xmu = x - x_mean
    xivar = np.sqrt(x_var + eps)
    xhat = xmu / xivar

    out = gamma * xhat + beta
    cache = (x, xmu, xivar, x_mean, xhat, gamma)
    return out, cache


def cs231n_groupnorm_backward(dy, cache):
    # Imitate batchnorm backward
    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    dx, dgamma, dbeta = None, None, None

    (x, xmu, xivar, x_mean, xhat, gamma) = cache
    N, H, W, G, C = x.shape
    reduce_size = H * W * C

    # step9
    dbeta = np.sum(dy, axis=(0, 1, 2), keepdims=True)
    dgammax = dy  # not necessary, but more understandable

    # step8
    dgamma = np.sum(dgammax*xhat, axis=(0, 1, 2), keepdims=True)
    dxhat = dgammax * gamma

    # step7
    divar = np.sum(dxhat*xmu, axis=(1, 2, 4), keepdims=True)
    dxmu1 = dxhat / xivar

    # step6
    dsqrtvar = -1. / (xivar**2) * divar

    # step5
    dvar = 0.5 * 1. / xivar * dsqrtvar

    # step4
    dsq = 1. / reduce_size * np.ones((N, H, W, G, C)) * dvar

    # step3
    dxmu2 = 2 * xmu * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=(1, 2, 4), keepdims=True)

    # step1
    dx2 = 1. / reduce_size * np.ones((N, H, W, G, C)) * dmu

    # step0
    dx = dx1 + dx2

    return dx, dgamma, dbeta


def rocking_groupnorm_backward(dy, cache):
    dx, dgamma, dbeta = None, None, None

    (x, xmu, xivar, x_mean, xhat, gamma) = cache
    N, H, W, G, C = x.shape
    reduce_size = H * W * C

    dbeta = np.sum(dy, axis=(0, 1, 2), keepdims=True)
    dgamma = np.sum(xhat * dy, axis=(0, 1, 2), keepdims=True)

    dlxhat = dy * gamma
    dxhatx = 1 / xivar

    dlvar = -0.5 * np.sum(dlxhat * xmu * xivar ** (-3),
                          axis=(1, 2, 4), keepdims=True)
    dvarx = 2 / reduce_size * \
        (xmu - 1 / reduce_size * np.sum(xmu, axis=(1, 2, 4), keepdims=True))

    dlmu = -np.sum(dlxhat / xivar, axis=(1, 2, 4), keepdims=True)
    dmux = 1 / reduce_size

    dx = dlxhat * dxhatx + dlvar * dvarx + dlmu * dmux

    return dx, dgamma, dbeta


def kevin_groupnorm_backward(dy, cache):
    # Imitate batchnorm backward
    # https://kevinzakka.github.io/2016/09/14/batch_normalization/
    dx, dgamma, dbeta = None, None, None

    (x, _, xivar, _, xhat, gamma) = cache
    N, H, W, G, C = x.shape
    reduce_size = H * W * C

    # intermediate partial derivatives
    dxhat = dy * gamma
    inv_var = 1./xivar

    # final partial derivatives
    dx = (1. / reduce_size) * inv_var * (reduce_size * dxhat - np.sum(dxhat, axis=(1, 2, 4), keepdims=True)
                                         - xhat * np.sum(dxhat * xhat, axis=(1, 2, 4), keepdims=True))
    dbeta = np.sum(dy, axis=(0, 1, 2), keepdims=True)
    dgamma = np.sum(xhat * dy, axis=(0, 1, 2), keepdims=True)

    return dx, dgamma, dbeta


def aten_cpu_groupnorm_backward(dy, cache):
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/group_norm_kernel.cpp#L655
    dx, dgamma, dbeta = None, None, None

    (x, xmu, xivar, x_mean, xhat, gamma) = cache
    rstd = 1 / xivar
    N, H, W, G, C = x.shape
    reduce_size = H * W * C

    # GroupNormBackward for dx
    ds = np.sum(dy * gamma * x, axis=(1, 2, 4), keepdims=True)
    db = np.sum(dy * gamma, axis=(1, 2, 4), keepdims=True)

    b = (db * x_mean - ds) * rstd ** (3) / reduce_size
    c = -b * x_mean - db * rstd / reduce_size
    dx = rstd * dy * gamma + b * x + c

    # GroupNormBackward for dgamma and dbeta
    dbeta = np.sum(dy, axis=(0, 1, 2), keepdims=True)
    dgamma = np.sum(xhat * dy, axis=(0, 1, 2), keepdims=True)

    return dx, dgamma, dbeta


if __name__ == '__main__':
    N = 1
    H = 2
    W = 2
    G = 2
    C = 2

    x = np.random.randn(N, H, W, G, C).astype(np.float32)
    gamma = np.random.randn(G, C).astype(np.float32)
    beta = np.random.randn(G, C).astype(np.float32)
    eps = 1e-5

    dy = np.random.randn(N, H, W, G, C).astype(np.float32)

    y, cache = groupnorm_forward(x, gamma, beta, eps)
    y1 = pytorch_groupnorm_forward(x, gamma, beta, eps)
    dx, dgamma, dbeta = pytorch_groupnorm_backward(x, gamma, beta, eps, dy)
    dx2, dgamma2, dbeta2 = cs231n_groupnorm_backward(dy, cache)
    dx3, dgamma3, dbeta3 = rocking_groupnorm_backward(dy, cache)
    dx4, dgamma4, dbeta4 = kevin_groupnorm_backward(dy, cache)
    dx5, dgamma5, dbeta5 = aten_cpu_groupnorm_backward(dy, cache)

    print('--------pytorch dx--------')
    print(dx.flatten())
    print('--------dx2--------')
    print(dx2.flatten())
    print('--------dx3--------')
    print(dx3.flatten())
    print('--------dx4--------')
    print(dx4.flatten())
    print('--------dx5--------')
    print(dx5.flatten())
