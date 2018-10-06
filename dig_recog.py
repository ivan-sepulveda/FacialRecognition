from pattern_recog_func import *
from sklearn.datasets import load_digits
dig_data = load_digits()
X = dig_data.data
X_img = dig_data.images
y = dig_data.target
x_range = X[:60]
y_range = y[:60]

md_train = svm_train(x_range,y_range)
misidentified_elements = 0

for i in range(60, 80):
    prediction = ((md_train.predict(X)[i]).reshape(1,-1)[0])
    answer = y[i]
    if prediction[0] != answer:
        misidentified_elements += 1
        print('index, actual digit, svm_predition', i, y[i], prediction[0])
success_rate = (20 - misidentified_elements)/(20)

"""Note for grader, please try running it in Notebook first, i've had better luck that way"""
print('Total number of mis-identifications {:d}'.format(misidentified_elements))
print('Succes rate: {:0.2%}'.format(success_rate))

# unseen_data = mpimage.imread('unseen_dig.png')
unseen_interpolated = interpol_im('unseen_dig.png', dim1 = 8, dim2 = 8, plot_new_im = True, axis_off = True)
plt.figure(figsize = (4,4))
plt.title('Image from data set')
plt.imshow(X_img[15], cmap = 'binary')
plt.grid('off')
plt.axis('off')

scaled_unseen_image = rescale_pixel(X, unseen_interpolated, ind = 15)

unseen_int_pre = md_train.predict(unseen_interpolated)
scaled_pre = md_train.predict(scaled_unseen_image)
print ('Prediction of unseen: {:d}'.format(unseen_int_pre[0]))
print('Prediction of scaled unseen: {:d}'.format(scaled_pre[0]))