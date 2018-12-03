# Python Notebook - Untitled Report

datasets

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset1 = datasets['Query 3']
df = pd.DataFrame(dataset1)
train_dataset = df.sample(frac=0.6, random_state=200)
test_dataset = df.drop(train_dataset.index)

train_dataset.head()

# Use the LABEL column in the train dataset as the training label
y_train = train_dataset.loc[ : , 'LABEL']
y_train = np.array(y_train)

# Remove LABEL and UUSERID in the train dataset, and use remaining columns as training features
X_train= train_dataset.drop(['LABEL', 'UUSERID'], 1)
X_train = np.array(X_train)

# Normalize the training features (each column of X) so that each column will have mean = 0 and standard deviation = 1.
X_train = StandardScaler().fit_transform(X_train)


# Use the LABEL column in the test dataset as the test label
y_test = test_dataset.loc[ : , 'LABEL' ]
y_test = np.array(y_test)

# Remove LABEL and UUSERID in the test dataset, and use remaining columns as test features
X_test= test_dataset.drop(['LABEL', 'UUSERID'], 1)
X_test = np.array(X_test)

# Normalize the test features (each column of X) so that each column will have mean = 0 and standard deviation = 1.
X_test = StandardScaler().fit_transform(X_test)

# Decomposit the nromalised traning features and project them to a 2 dimensional space
pca = PCA(n_components=2).fit(X_train)
pca_2d_train = pca.transform(X_train)

# Create scatter plot
for i in range(0, pca_2d_train.shape[0]):
  if y_train[i] == 0:
    c1 = plt.scatter(pca_2d_train[i,0],pca_2d_train[i,1], c='r', marker='+')
  elif y_train[i] == 1:  
    c2 = plt.scatter(pca_2d_train[i,0],pca_2d_train[i,1], c='g', marker='o')
    
plt.legend([c1, c2], ['Buyer', 'Renter'])
plt.title('Training dataset with 2 classes and known outcomes')
plt.show()


# Decomposit the nromalised test features and project them to a 2 dimensional space
pca = PCA(n_components=2).fit(X_test)
pca_2d_test = pca.transform(X_test)

# Create scatter plot
plt.scatter(pca_2d_test[:,0], pca_2d_test[:,1])
plt.title('Test dataset with unknown 2 classes')


cm_bright = ListedColormap(['g', 'r'])
cm_dark = ListedColormap(['b', 'm'])
    
ax = plt.subplot(1, 1, 1)
# Plot the training points
ax.scatter(pca_2d_train[:, 0], pca_2d_train[:, 1], c=y_train, cmap=cm_bright, marker='o', label='train')
# and testing points
ax.scatter(pca_2d_test[:, 0], pca_2d_test[:, 1], c=y_test, cmap=cm_dark, marker='+', label='test')

ax.legend(loc="upper right")
ax.set_title('Train and Test data with known 2 classes')
plt.show()


clf = svm.SVC(kernel='linear')

# Fit the SVM model according to the given training data.
clf.fit(X_train,y_train)


# After being fitted, the model can then be used to predict new values:
predicted = clf.predict(X_test)
print(predicted)


# Returns the mean accuracy on the given test data and labels.
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))


plt.subplot(1,1,1)
plt.title('Test data, with errors highlighted')
colors=['r','g']

for t in [0,1]:
    plt.plot(pca_2d_test[y_test==t][:,0],pca_2d_test[y_test==t][:,1],colors[t]+'*')

errX,errY=pca_2d_test[predicted!=y_test],y_test[predicted!=y_test]
for t in [0,1]:
    plt.plot(errX[errY==t][:,0],errX[errY==t][:,1],colors[t]+'o')


h = .02  # step size in the mesh

names = ["Linear SVM", "RBF SVM", "Poly SVM", "Sigmoid SVM"]
classifiers = [svm.SVC(kernel="linear", C=1),
               svm.SVC(kernel='rbf', gamma=0.05, C=1),
               svm.SVC(kernel="poly", C=1),
               svm.SVC(kernel="sigmoid", gamma=0.5)]

# Randomly sample 20% of your dataframe
exp_dataset = df.sample(frac=0.2,random_state=200)

# Use the LABEL column in the test dataset as the test label
y_exp = exp_dataset.loc[ : , 'LABEL' ]
y_exp = np.array(y_exp)

# Remove LABEL and UUSERID in the test dataset, and use remaining columns as test features
X_exp= exp_dataset.drop(['LABEL', 'UUSERID'], 1)
X_exp = np.array(X_exp)
X_exp = StandardScaler().fit_transform(X_exp)
pca = PCA(n_components=2).fit(X_exp)
X_exp = pca.transform(X_exp)

datasets = [(X_exp, y_exp)]


figure = plt.figure(figsize=(18, 6))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#008000', '#FF4500'])
    cm_dark = ListedColormap(['#008000', '#FF4500'])
    
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_dark)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_dark)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()




