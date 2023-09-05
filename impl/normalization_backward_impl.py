import numpy as np
import torch


def NHWGC2NCHW(x):
    N, H, W, G, C1 = x.shape
    C = G * C1
    x = x.reshape(N, H, W, C)
    return np.transpose(x, (0, 3, 1, 2))


def NCHW2NHWGC(x, G):
    N, C, H, W = x.shape
    C1 = C // G
    return np.transpose(x, (0, 2, 3, 1)).reshape(N, H, W, G, C1)


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


def normalization_forward(x, gamma, beta, eps, reduce_axis):
    x_mean = np.mean(x, axis=reduce_axis, keepdims=True)
    x_var = np.var(x, axis=reduce_axis, keepdims=True)

    rstd = 1 / np.sqrt(x_var + eps)
    xhat = (x - x_mean) * rstd

    out = gamma * xhat + beta
    return out, x_mean, rstd


def normalization_input_backward(dy, x, gamma, x_mean, rstd, reduce_axis, reduce_size):
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L196
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/group_norm_kernel.cpp#L655
    ds = np.sum(dy * gamma * x, axis=reduce_axis, keepdims=True)
    db = np.sum(dy * gamma, axis=reduce_axis, keepdims=True)

    b = (db * x_mean - ds) * rstd ** (3) / reduce_size
    c = -b * x_mean - db * rstd / reduce_size
    dx = rstd * dy * gamma + b * x + c

    return dx


def normalization_gamma_beta_backward(dy, x, x_mean, rstd, reduce_axis):
    # Assume shape of gamma and beta are the same
    dgamma = np.sum(dy * (x - x_mean) * rstd, axis=reduce_axis, keepdims=True)
    dbeta = np.sum(dy, axis=reduce_axis, keepdims=True)

    return dgamma, dbeta


def layernorm_forward(x, gamma, beta, eps):
    # x = [M, K], gamma, beta = [1, K], reduce_axis = 1
    return normalization_forward(x, gamma, beta, eps, 1)


def layernorm_backward(dy, x, gamma, x_mean, rstd):
    # dy, x = [M, K], gamma = [1, K], x_mean, rstd = [M, 1]
    M, K = x.shape
    dx = normalization_input_backward(dy, x, gamma, x_mean, rstd, 1, K)
    dgamma, dbeta = normalization_gamma_beta_backward(dy, x, x_mean, rstd, 0)
    return dx, dgamma, dbeta


def groupnorm_forward(x, gamma, beta, eps):
    # x = [N, H, W, G, C], gamma, beta = [1, 1, 1, G, C], reduce_axis = (1, 2, 4)
    return normalization_forward(x, gamma, beta, eps, (1, 2, 4))


def groupnorm_backward(dy, x, gamma, x_mean, rstd):
    # dy, x = [N, H, W, G, C], gamma = [1, 1, 1, G, C], x_mean, rstd = [N, 1, 1, G, 1]
    N, H, W, G, C = x.shape
    dx = normalization_input_backward(
        dy, x, gamma, x_mean, rstd, (1, 2, 4), H * W * C)
    dgamma, dbeta = normalization_gamma_beta_backward(
        dy, x, x_mean, rstd, (0, 1, 2))
    return dx, dgamma, dbeta


def verify(golden, x, tol=1e-5):
    err = abs(golden - x)
    if np.any(err > tol):
        print('Fail')
    else:
        print('Pass')


def layernorm_test():
    M = 10
    K = 50
    x = np.random.randn(M, K).astype(np.float32)
    gamma = np.random.randn(K).astype(np.float32)
    beta = np.random.randn(K).astype(np.float32)
    dy = np.random.randn(M, K).astype(np.float32)
    eps = 1e-5

    y, x_mean, rstd = layernorm_forward(x, gamma, beta, eps)
    dx, dgamma, dbeta = layernorm_backward(dy, x, gamma, x_mean, rstd)
    golden_dx, golden_dgamma, golden_dbeta = pytorch_layernorm_backward(
        x, gamma, beta, eps, dy)

    verify(golden_dx, dx)
    verify(golden_dgamma, dgamma)
    verify(golden_dbeta, dbeta)


def groupnorm_test():
    N, H, W, G, C = 2, 16, 16, 8, 32
    x = np.random.randn(N, H, W, G, C).astype(np.float32)
    gamma = np.random.randn(G, C).astype(np.float32)
    beta = np.random.randn(G, C).astype(np.float32)
    dy = np.random.randn(N, H, W, G, C).astype(np.float32)
    eps = 1e-5

    y, x_mean, rstd = groupnorm_forward(x, gamma, beta, eps)
    dx, dgamma, dbeta = groupnorm_backward(dy, x, gamma, x_mean, rstd)
    golden_dx, golden_dgamma, golden_dbeta = pytorch_groupnorm_backward(
        x, gamma, beta, eps, dy)

    verify(golden_dx, dx)
    verify(golden_dgamma, dgamma, 1e-4)
    verify(golden_dbeta, dbeta, 1e-4)


if __name__ == '__main__':
    layernorm_test()
    groupnorm_test()
