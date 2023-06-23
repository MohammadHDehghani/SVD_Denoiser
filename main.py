import numpy as np
from PIL import Image
import math
import random
from matplotlib import pyplot as plt


# Hyperparameters
mean = 30
std = 80


def generate_normal(mean, std):
    u1 = random.random()
    u2 = random.random()
    r = math.sqrt(-2 * math.log(u1))
    theta = 2 * math.pi * u2
    z = r * math.cos(theta)
    return mean + z * std


def g_noiser(input_image, mean, std):
    shape = input_image.shape
    noise = np.zeros(shape)
    for i in range(3):
        for j in range(shape[0]):
            for k in range(shape[1]):
                noise[j, k, i] = generate_normal(mean, std)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def eig(A):
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")

    if not np.allclose(A, A.T.conj()):
        raise ValueError("Matrix must be Hermitian or real symmetric")

    eigvals = np.zeros(n)
    eigvecs = np.eye(n)

    for i in range(n):
        eigval, eigvec = power_method(A, eigvecs[:, i])

        eigvals[i] = eigval
        eigvecs[:, i] = eigvec

        A = A - eigval * np.outer(eigvec, eigvec)

    return eigvals, eigvecs


def power_method(A, x0, tol=1e-8, maxiter=1000):
    x = x0 / np.linalg.norm(x0)

    for i in range(maxiter):
        Ax = A.dot(x)

        x_new = Ax / np.linalg.norm(Ax)

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    eigval = x.T.dot(A).dot(x)

    return eigval, x


def svd(A):
    ATA = A.T.dot(A)
    eigvals, eigvecs = eig(ATA)

    S = np.sqrt(eigvals)

    sort_indices = np.argsort(S)[::-1]
    S = S[sort_indices]

    Vt = eigvecs[:, sort_indices]

    U = A.dot(Vt) / S[np.newaxis, :]

    return U, S, Vt.T


def svd_denoiser(noisy_image):
    img = noisy_image.astype(float) / 255.0

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    Ur, sr, Vr = svd(R)
    Ug, sg, Vg = svd(G)
    Ub, sb, Vb = svd(B)

    threshold = 3

    sr_thresh = np.where(sr < threshold, 0, sr - threshold)
    sg_thresh = np.where(sg < threshold, 0, sg - threshold)
    sb_thresh = np.where(sb < threshold, 0, sb - threshold)

    R_denoised = Ur.dot(np.diag(sr_thresh)).dot(Vr)
    G_denoised = Ug.dot(np.diag(sg_thresh)).dot(Vg)
    B_denoised = Ub.dot(np.diag(sb_thresh)).dot(Vb)

    img_denoised = np.stack([R_denoised, G_denoised, B_denoised], axis=2)

    img_denoised = (img_denoised * 255.0).astype(np.uint8)

    return img_denoised


if __name__ == '__main__':
    image = Image.open(r'./images/10111575845_1b49137d9d_n.jpg')
    image = np.asarray(image)
    noisy_image = g_noiser(image, mean, std)
    denoised_image = svd_denoiser(noisy_image)
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 4, 2)
    plt.imshow(noisy_image)
    plt.title('Noisy Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 4, 3)
    plt.imshow(denoised_image)
    plt.title('DeNoised Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()
