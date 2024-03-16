# Aluno: Hiago dos Santos Rabelo - 160124492

import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_contours(img, contours):
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow('contours image', img)

def filter_image(img):
    kernel = np.ones((15,15),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    proc_img = cv2.medianBlur(opening, 19)
    cv2.imshow('processed img', proc_img)
    cv2.imwrite('imgs/processed_brain.jpg', proc_img)

    return proc_img

def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('loaded img', img)
    return img

def get_histogram(img):
    hist_proc = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.plot(hist_proc)
    plt.savefig(f'imgs/hist.png')
    plt.close()

def get_histogram_raw(img):
    hist_proc = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.plot(hist_proc)
    plt.savefig(f'imgs/raw_hist.png')
    plt.close()

def binarize(img):
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow('mask', thresh)
    cv2.imwrite('imgs/thresh_brain.jpg', thresh)
    return thresh

def largest_area(mask, original_img):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_item= sorted_contours[0]
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(original_img, largest_item, -1, (255,0,0), 2)
    cv2.imshow('contour', original_img)
    cv2.imwrite('imgs/contour_brain.jpg', original_img)

def questao1():
    img_path = 'imgs/brain.jpg'
    
    img = load_image(img_path)
    get_histogram_raw(img)
    processed_image = filter_image(img)
    get_histogram(processed_image)
    mask = binarize(processed_image)
    largest_area(mask, img)

    cv2.waitKey()

def questao2():
    # Code based on https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    img_path = '1.jpeg'
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2YUV)
    img = cv2.medianBlur(raw_img, 33)
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    z = img.reshape((-1, 3))
    z = img.reshape((-1, 3))
    z = np.float32(z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
    k = 3
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('res', res2)
    cv2.imwrite('imgs/result_vegetables_seg.jpg', res2)
    cv2.imshow('filtered', img)
    cv2.imwrite('imgs/median_vegetables.jpg', img)
    cv2.imshow('original1', cv2.cvtColor(raw_img, cv2.COLOR_YUV2BGR))
    cv2.waitKey()

if __name__ == '__main__':
    # questao1()
    questao2()