from PyQt5.QtWidgets import QCheckBox, QWidget
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from feature_selection import *
from graphics import show_features, show_features_progress


class Main(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.resize(300, 100)

        self.isdatasetloaded = QCheckBox("Загрузить датасет", self)
        self.isdatasetloaded.move(10, 10)

        self.isfeaturesgot = QCheckBox("Получение признаков", self)
        self.isfeaturesgot.move(10, 30)

        self.iscrossvalidation = QCheckBox("Кросс-Валидация", self)
        self.iscrossvalidation.move(10, 50)

        self.show()
        self.work_process()

    def work_process(self):
        data_images = fetch_olivetti_faces()
        images = data_images['images']
        target = data_images['target']
        classes = np.unique(target)[-1]
        show_features(images[0])
        self.isdatasetloaded.setChecked(True)

        features_data = []
        functions = [histogram, dft, dct, mean_pooling, gradient]
        features_lens = [len(histogram(images[0])), len(dft(images[0]).flatten()), len(dct(images[0]).flatten()),
                            len(mean_pooling(images[0]).flatten()), len(gradient(images[0]))]
        for f, n in zip(functions, features_lens):
            features_data.append(make_dataset(images, f, n))
        self.isfeaturesgot.setChecked(True)

        fig, axs = show_features_progress()
        rez = np.zeros((5, 9))     
        for k in range(1, 10):
            centers = [] 
            train_data = []
            train_target = []

            for feature in range(5):
                centers.append([])
                train_data.append([])
                for c in range(classes):
                    centers[feature].append(np.mean(features_data[feature][c*10 : c*10 + k], axis=0))
                    if len(train_data[feature]) > 0:
                        train_data[feature] = np.concatenate((train_data[feature], features_data[feature][c*10 + k:(c+1)*10]))
                        if feature == 0:
                            train_target = np.concatenate((train_target, target[c*10 + k:(c+1)*10]))
                    else:
                        for arr in features_data[feature][c*10 + k:(c+1)*10]:
                            train_data[feature].append(arr)
                        if feature == 0:
                            train_target = target[c*10 + k:(c+1)*10]
            train_target = np.array(train_target)
            loss = [np.sum(target_by_centers(train_data[feature], centers[feature]) != train_target)/len(train_target) for feature in range(5)]
            print(loss)
            rez[:, k-1] = loss
            print(rez)
            n = 0
            for i in range(2):
                for j in range(3):
                    if i == 1 and j == 2:
                        break
                    axs[i, j].plot(rez[n, :k])
                    n+=1
                    fig.show()
                plt.pause(0.05)
        print('11')
        self.iscrossvalidation.setChecked(True)
