from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.image as mpimage
from scipy.interpolate import interp2d
from sklearn import svm
from sklearn import preprocessing

def interpol_im(image_file, dim1=8, dim2=8, plot_new_im=False, cmap='binary', axis_off=False):
    image_data = mpimage.imread(image_file)
    # Now lets disregard the irrelevant channels
    image_data = image_data[:, :, 0]

    x = np.arange(image_data.shape[1])
    y = np.arange(image_data.shape[0])

    alpha = interp2d(x, y, image_data)

    x_new = np.linspace(0, image_data.shape[1], dim1)
    y_new = np.linspace(0, image_data.shape[0], dim2)

    interpolated_image_data = alpha(x_new, y_new)

    if plot_new_im == True:
        plt.imshow(interpolated_image_data, cmap=cmap, interpolation='nearest')
        if axis_off == True:
            plt.axis('off')
        plt.grid('off')
        plt.title('Interpolated Image')
        plt.show()
    flattened_interpolated_image_data = interpolated_image_data.flatten()
    return flattened_interpolated_image_data

def pca_X(X, n_comp = 50):
    md_pca = PCA(n_comp, whiten = True)
    Xproj = md_pca.fit_transform(X)
    return md_pca, Xproj


def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60):
    flattened = interpol_im(imfile, dim1, dim2, plot_new_im = True)
    Xproj = md_pca.transform(flattened)
    prediction = md_clf.predict(Xproj[0])
    return prediction

def rescale_pixel(X, unseen, ind = 0):
    scaler = preprocessing.MinMaxScaler(feature_range = (min(X[ind]), max(X[ind])))
    scaled_unseen = scaler.fit_transform(X[ind],unseen).astype(int)
    return scaled_unseen

def svm_train(X, y, gamma = 0.001, C = 100):
    clf = svm.SVC(kernel = "poly", gamma = gamma, C = C)
    md_clf = clf.fit(X, y)
    return md_clf