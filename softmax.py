import numpy as np
import torch


def pytorch_softmax_fwd(x, dim=0):
    softmax = torch.nn.Softmax(dim=dim)

    tx = torch.from_numpy(x)
    ty = softmax.forward(tx)

    return ty.numpy()


def pytorch_softmax_bwd(x, dy, dim=0):
    softmax = torch.nn.Softmax(dim=dim)

    tx = torch.from_numpy(x)
    tx.requires_grad = True
    ty = softmax.forward(tx)
    tdy = torch.from_numpy(dy)
    ty.backward(tdy)

    return tx.grad.numpy()


def softmax_fwd(x, dim=0):
    ex = np.exp(x - np.max(x, axis=dim))
    return ex / ex.sum(axis=dim)


def softmax_bwd(x, dy, dim=0):
    y = softmax_fwd(x, dim)

    dx = np.diag(y)-np.outer(y, y)
    return np.dot(dx, dy)


if __name__ == '__main__':
    x = np.random.randn(10).astype(np.float32)
    dy = np.random.randn(10).astype(np.float32)

    y_golden = pytorch_softmax_fwd(x)
    y = softmax_fwd(x)

    dx_golden = pytorch_softmax_bwd(x, dy)
    dx = softmax_bwd(x, dy)

    print("y_golden: ", y_golden)
    print("y: ", y)
    print("dx_golden: ", dx_golden)
    print("dx: ", dx)
