import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from sklearn import model_selection
from skimage.feature import hog
from sklearn import metrics


def read_img(path, n=25 ** 2):
    """
    Read image and store it as an array, given the image path.
    Returns the 3 dimensional image array.
    """
    img = Image.open(path).convert('L')
    img = img.resize((n, n), Image.ANTIALIAS)
    img_arr = np.array(img, dtype='int32')
    img.close()
    return img_arr


def reshapemat(v):
    da = int(np.sqrt(len(v)))
    return np.reshape(v, (da, da)).T


def testModel(pred, test):
    P, R, F, _ = metrics.precision_recall_fscore_support(test, pred)
    print("  Precision")
    print(P)
    print("  Recall:")
    print(R)
    print("  F-1 Score:")
    print(F)
    print("---------------------------------------------------------------------")

###### Load Data #####
show9 = True
if show9:
    # show 9 fails
    for i in range(9):
        plt.subplot(330 + 1 + i)
        fn = 'data/prints/fail/failpic' + str(i) + '.jpg'
        image = read_img(fn)
        plt.imshow(image)
    plt.show()
    # show 9 passes
    for i in range(9):
        plt.subplot(330 + 1 + i)
        fn = 'data/prints/okay/okaypic' + str(i) + '.jpg'
        image = read_img(fn)
        plt.imshow(image)
    plt.show()

dirf = 'data/prints/fail/'
dirp = 'data/prints/okay/'

images = []
labels = []
hogs = []
hogimage = []

#fails
for file in listdir(dirf):
    im = read_img(dirf + file)
    h, him = hog(im, visualize=True)
    hogs.append(h)
    hogimage.append(him)
    images.append(im)
    labels.append(1)

#passes
for file in listdir(dirp):
    im = read_img(dirp + file)
    h, him = hog(im, visualize=True)
    hogs.append(h)
    hogimage.append(him)
    images.append(im)
    labels.append(0)

showhogs = True
if showhogs:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Histogram of Oriented Gradients (hog)")
    ax[0].imshow(hogimage[-1])
    ax[0].set_title("Passed Example")
    ax[1].imshow(hogimage[0])
    ax[1].set_title("Failed Example")
    plt.show()

print("Data Loaded Properly")




##### Run Model #####
X = np.asarray(hogs)
y = labels
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=53)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.fit(X_train, y_train).predict(X_test)

print(metrics.classification_report(y_test, y_pred))


metrics.accuracy_score(y_test, y_pred)