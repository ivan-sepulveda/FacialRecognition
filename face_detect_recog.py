from __future__ import division
from pattern_recog_func import *
import cv2

all_student_faces_list = []
y = []
names_dict = {0: 'Gilbert', 1: 'Janet'}

for a in range(40):
    file_name_a = "Gilbert_" + str(a) + ".png"
    interpolated_gilbert = interpol_im(file_name_a, dim1=45, dim2=60)
    all_student_faces_list.append(interpolated_gilbert)
    y.append(0)
for b in range(39):
    file_name_b = "Janet_" + str(b) + ".png"
    interpolated_janet = interpol_im(file_name_b, dim1=45, dim2=60)
    all_student_faces_list.append(interpolated_janet)
    y.append(1)
# for c in range(40):
#     file_name_c = "Luke_" + str(b) + ".png"
#     interpolated_luke = interpol_im(file_name_c, dim1 = 45, dim2 = 60)
#     all_student_faces_list.append(interpolated_luke)
#     y.append(2)

y_array = np.array(y)

all_student_faces_list = np.array(all_student_faces_list)
X = np.vstack(all_student_faces_list)

correct = 0
for z in range(len(y)):
    Xtrain = np.delete(X, z, axis=0)
    ytrain = np.delete(y, z)
    Xcheck = X[z].reshape(1, -1)
    md_pca, Xproj = pca_X(Xtrain, 50)

    Xtrain_proj = md_pca.transform(Xtrain)
    Xcheck_proj = md_pca.transform(Xcheck)

    md_train = svm_train(Xtrain_proj, ytrain)
    answer = md_train.predict(Xcheck_proj)[0]
    print("Prediected was: {}".format(names_dict[answer]))
    print("Correct was: {}".format(names_dict[y[z]]))
    print("\n")
    if names_dict[answer] == names_dict[y[z]]:
        correct += 1
success = correct / (len(all_student_faces_list))
print('Succes rate: {:0.2%}'.format(success))

imagePath = 'whoswho.jpg'
cascPath = 'haarcascade_frontalface_default.xml'
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
# Read the image
image = mpimage.imread(imagePath)
# This is still just a 2d array -- except it's set in a certain kind of gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(200, 200),
    flags=cv2.CASCADE_SCALE_IMAGE
)

t = 0
for (x, y, w, h) in faces:
    t = t + 1
    mpimage.imsave("person{}.jpeg".format(str(t)), image[y:y + h, x:x + w], vmin=None, vmax=None, cmap=None,
                   format=None, origin=None, dpi=100)
# cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 6)
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.grid('off')
plt.axis('off')
plt.show()

person1 = pca_svm_pred("person1.jpeg", md_pca, md_train)
person2 = pca_svm_pred("person2.jpeg", md_pca, md_train)

i1 = mpimage.imread("person1.jpeg")
i2 = mpimage.imread("person2.jpeg")

print('PCA+SVM prediction for person 1: ', names_dict[int(person1)])
plt.imshow(i2)
print('PCA+SVM prediction for person 2: ', names_dict[int(person2)])
