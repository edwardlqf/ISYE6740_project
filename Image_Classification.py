import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from sklearn import model_selection
from skimage.feature import hog
from sklearn import metrics
import os
from random import randint
from sklearn.linear_model import SGDClassifier
import pickle


def read_img(path, n=10**2):
    """
    Read image and store it as an array, given the image path.
    Returns the 3 dimensional image array.
    """
    img = Image.open(path).convert('L')
    img = img.resize((n, n), Image.ANTIALIAS)
    img_arr = np.array(img, dtype='int32')
    img.close()
    return img_arr

##### Adjustable Parameters #####
ppc = 12
cpb = 2
n = 16**2
rint = randint(0, 100)

###### Load Data #####
dirf = 'data/prints/fail/'
dirp = 'data/prints/okay/'
nump = len(os.listdir(dirp))
numf = len(os.listdir(dirp))
numtot = nump + numf

images = []
labels = []
hogs = []
hogimage = []
i = 0
# fails
for file in listdir(dirf):
    im = read_img(dirf + file, n=n)
    h, him = hog(im, visualize=True,
                 pixels_per_cell=(ppc, ppc),
                 cells_per_block=(cpb, cpb),
                 block_norm="L2-Hys")
    hogs.append(h)
    hogimage.append(him)
    images.append(im)
    labels.append(1)
    if i % 12 == 0:
        print(i / numtot * 100, "%")
    i += 1

# passes
for file in listdir(dirp):
    im = read_img(dirp + file, n=n)
    h, him = hog(im, visualize=True,
                 pixels_per_cell=(ppc, ppc),
                 cells_per_block=(cpb, cpb),
                 block_norm="L2-Hys")
    hogs.append(h)
    hogimage.append(him)
    images.append(im)
    labels.append(0)
    if i % 12 == 0:
        print(i / numtot * 100, "%")
    i += 1

showhogs = True
if showhogs:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Histogram of Oriented Gradients (hog)")
    ax[0].imshow(hogimage[-1])
    ax[0].set_title("Passed Example")
    ax[1].imshow(hogimage[0])
    ax[1].set_title("Failed Example")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.show()

show9 = True
if show9:
    plt.text(.5, .975, "txt", ha='center')
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.axis('off')
        im = images[i]
        plt.imshow(im)
    plt.show()
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.axis('off')
        im = hogimage[i]
        plt.imshow(im)
    plt.show()
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.axis('off')
        im = images[-i - 1]
        plt.imshow(im)
    plt.show()
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.axis('off')
        im = hogimage[-i - 1]
        plt.imshow(im)
    plt.axis('off')
    plt.show()
print("Preprocessing complete")

##### Run Model #####
X = np.array(hogs)
y = labels
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=rint)

sgd_clf = SGDClassifier(random_state=rint, max_iter=1000, tol=1e-4, penalty='l2', loss="log")
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

pickle.dump(sgd_clf, open("SGDModel.pkl", 'wb'))

##### Make a Prediction #####
dothis = False
if dothis:
    predlabels = [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    dirpred = 'data/prints/prediction/'
    predimages = []
    predhogs = []
    predhogimages = []

    for file in listdir(dirpred):
        im = read_img(dirpred + file, n=n)
        h, him = hog(im, visualize=True,
                     pixels_per_cell=(ppc, ppc),
                     cells_per_block=(cpb, cpb),
                     block_norm="L2-Hys")
        predhogs.append(h)
        predhogimages.append(him)
        predimages.append(im)
    X_predict = np.array(predhogs)
    predictions = sgd_clf.predict(X_predict)
    probabilities = sgd_clf.predict_proba(X_predict) * 100


def makePred(path, model, figures=True, ppc=12, cpb=2, n=16**2):
    import numpy as np
    from PIL import Image
    from skimage.feature import hog

    imgin = Image.open(path)
    img = imgin.convert('L').resize((n, n), Image.ANTIALIAS)
    img_arr = np.array(img, dtype='int32')
    ho, hogimage = hog(img_arr, visualize=True,
                       pixels_per_cell=(ppc, ppc),
                       cells_per_block=(cpb, cpb),
                       block_norm="L2-Hys")
    predin = ho.reshape(1, -1)
    pred = model.predict(predin)
    pred = pred[0]
    prob = model.predict_proba(predin)
    if figures:
        import matplotlib.pyplot as plt
        img_arr2 = np.array(imgin, dtype='int32')
        fig, ax = plt.subplots()
        ax.imshow(img_arr2)
        ax.set_xticks([])
        ax.set_yticks([])
        if pred == 0:
            txt = "Classified as an okay print with " + str(round(prob.max(), 2) * 100) + "% probability"
        elif pred == 1:
            txt = "Classified as a failed print with " + str(round(prob.max(), 2) * 100) + "% probability"
        fig.text(.5, .975, txt, ha='center')
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=2)
        fig.suptitle("Image Processing Steps")
        ax[0].imshow(img_arr)
        ax[0].set_title("Converted Image")
        ax[1].imshow(hogimage)
        ax[1].set_title("HOG Image")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()
    img.close()
    return pred, prob.max()


path = 'data/prints/prediction/prediction03.jpg'

mod = pickle.load(open("SGDModel.pkl", 'rb'))
makePred(path, model=mod, ppc=ppc, cpb=cpb, n=n)
