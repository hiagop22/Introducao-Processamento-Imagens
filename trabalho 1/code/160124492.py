from matplotlib import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv

def downscale(img: np.ndarray, factor: float):
    if not factor or factor >= 1:
        return img

    h_mask = np.ones(img.shape[0], dtype=np.bool8)
    w_mask = np.ones(img.shape[1], dtype=np.bool8)
    
    step = int(1/factor)
    h_mask[::step] = False
    w_mask[::step] = False

    return img[h_mask,:][:, w_mask]

def upscale(img: np.ndarray, factor: float):
    if factor <= 1:
        return img

    new_h_pos = (np.array(range(img.shape[0]))*factor).astype(int)
    new_w_pos = (np.array(range(img.shape[1]))*factor).astype(int)

    new_shape = (np.array(img.shape)*factor).astype(int)
    new_img = np.zeros(new_shape, dtype=np.uint)

    for i, h_pos in enumerate(new_h_pos):
        new_img[h_pos, new_w_pos] = img[i, :]

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            if i not in new_h_pos or j not in new_w_pos:
                dist_x = abs(new_h_pos - i).argmin()
                dist_y = abs(new_w_pos - j).argmin()
                new_img[i, j] = new_img[new_h_pos[dist_x], new_w_pos[dist_y]]
    return new_img

def questao1():
    down_factors = (0.5, 0.2)
    up_factors = (1.4, 2.0)

    img = np.array(Image.open('imgs/notre-dame.pgm').convert('L'), dtype=np.uint8)
    # downscaled = [downscale(img, factor) for factor in down_factors]
    # upscaled = [upscale(img, factor) for factor in up_factors]

    plt.imshow(img, cmap='gray')
    plt.savefig(f'figs_trabalho/notre-dame.png')

    # for i, img in enumerate(downscaled):
    #     plt.imshow(img, cmap='gray')
    #     plt.savefig(f'figs_trabalho/downscaled_{i}.png')

    # for i, img in enumerate(upscaled):
    #     plt.imshow(img, cmap='gray')
    #     plt.savefig(f'figs_trabalho/upscaled_{i}.png')
    
def homomorphic(complex: np.ndarray):
    yh = 4
    yl = 0.2
    d0 = 5

    P = complex.shape[0]/2
    Q = complex.shape[1]/2
    H = np.zeros(complex.shape)
    U, V = np.meshgrid(range(complex.shape[0]), range(complex.shape[1]), sparse=False, indexing='ij')
    Duv = (((U-P)**2+(V-Q)**2)).astype(float)
    re = np.exp(-Duv/(d0**2))
    h = (yh - yl) * (1 - re) + yl

    filtered = complex * h
    
    filtered = np.fft.ifftshift(filtered)
    filtered = cv.idft(filtered)

    cv.normalize(filtered, filtered, 0, 1, cv.NORM_MINMAX)
    filtered = np.exp(filtered)*np.exp(0.1)

    plt.imshow(filtered, cmap='gray')
    plt.savefig('figs_trabalho/homomorfico.png')

def questao2():
    # based on 
    # https://docs.opencv.org/4.x/d8/d01/tutorial_discrete_fourier_transform.html
    # https://medium.com/@elvisdias/introduction-to-fourier-transform-with-opencv-922a79cddf36

    img = np.array(Image.open('imgs/image2.jpg').convert('L'), dtype=np.uint8)
    padded = np.log(img + 0.1)
    complex = cv.dft(np.float32(padded)/255.0)
    complex = np.fft.fftshift(complex)
    homomorphic(complex)
    plt.imshow(img, cmap='gray')
    plt.savefig('figs_trabalho/original_homomorphic.png')

def butterwoth(d0, shape: tuple, u_k: int=0, v_k: int=0, n: int=4):
    p = shape[0]/2
    q = shape[1]/2
    u, v = np.meshgrid(range(shape[0]), range(shape[1]), sparse=False, indexing='ij')
    d = (((u-p-u_k)**2+(v-q-v_k)**2)**0.5).astype(float)
    filter = (1 + (d/d0)**(2*n))**(-1)

    return filter

def questao3():
    # based on
    # https://github.com/dushyant18033/Digital-Image-Processing/blob/main/2018033_DIP_A3/P_DUSHYANT_2018033/DUSHYANT_2018033_code.py

    img = np.array(Image.open('imgs/moire.tif').convert('L'), dtype=np.uint8)
    plt.imshow(img, cmap='gray')
    plt.savefig('figs_trabalho/notch_original.png')
    shape = img.shape

    d0_u_v = [  (10,39,55), 
                (10,-39,55),
                (5,78,55),
                (5,-78,55)]

    filter = np.ones_like(img, dtype=np.float32)

    for d0, u_k, v_k in d0_u_v:
        filter *= 1-butterwoth(d0=d0, shape=shape, u_k=u_k, v_k=v_k)

    fft_img = cv.dft(np.float32(img))
    fft_img = np.fft.fftshift(fft_img)
    plt.imshow(np.log(np.abs(fft_img) + 0.1), cmap='gray')
    plt.savefig('figs_trabalho/fft_img.png')

    fft_img*= filter
    plt.imshow(np.log(np.abs(fft_img) + 0.1), cmap='gray')
    plt.savefig('figs_trabalho/notch_filter.png')
    
    filtered = np.fft.ifftshift(fft_img)
    filtered = cv.idft(filtered)
    cv.normalize(filtered, filtered, 0, 1, cv.NORM_MINMAX)
    plt.imshow(filtered, cmap='gray')
    plt.savefig('figs_trabalho/notched.png')

def main():
    questao1()
    # questao2()
    # questao3()

if __name__ == '__main__':
    main()