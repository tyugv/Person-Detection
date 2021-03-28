from PyQt5.QtWidgets import QCheckBox, QWidget, QTableWidget, QTableWidgetItem, QLabel
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neighbors import KNeighborsClassifier as KNN
from feature_selection import *
from graphics import show_features, show_features_progress


class Main(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.resize(430, 350)
        self.move(900, 200)

        self.isdatasetloaded = QCheckBox("Загрузить датасет", self)
        self.isdatasetloaded.move(10, 10)

        self.isfeaturesgot = QCheckBox("Получение признаков", self)
        self.isfeaturesgot.move(10, 30)

        self.iscrossvalidation = QCheckBox("Кросс-Валидация", self)
        self.iscrossvalidation.move(10, 50)

        self.table_label = QLabel(self)
        self.table_label.setText("Лучшие результаты классификации")
        self.table_label.setGeometry(130, 80, 200, 20)

        self.table = QTableWidget(6, 4, self)
        self.table.setGeometry(10, 100, 400, 250)
        feature_names = ['Градиент', 'Scale', 'DCT', 'DFT', 'Гистограмма']
        column_names = ['Признак', 'Правильно', 'Неправильно', 'Точность']
        for c in range(4):
            item = QTableWidgetItem(column_names[c])
            self.table.setItem(0, c, item)
        for r in range(5):
            item = QTableWidgetItem(feature_names[r])
            self.table.setItem(r + 1, 0, item)

        self.show()
        #self.work_process()

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
            train_data = []
            train_target = []
            test_data = []
            test_target = []

            for feature in range(5):
                train_data.append([])
                test_data.append([])
                for c in range(classes):
                    if len(train_data[feature]) > 0:
                        train_data[feature] = np.concatenate((train_data[feature], features_data[feature][c*10 + k:(c+1)*10]))
                        test_data[feature] = np.concatenate((test_data[feature], features_data[feature][c*10 : c*10 + k]))
                        if feature == 0:
                            train_target = np.concatenate((train_target, target[c*10 + k:(c+1)*10]))
                            test_target = np.concatenate((test_target, target[c*10 : c*10 + k]))
                    else:
                        for arr in features_data[feature][c*10 + k:(c+1)*10]:
                            train_data[feature].append(arr)
                        for arr in features_data[feature][c*10 : c*10 + k]:
                            test_data[feature].append(arr)
                        if feature == 0:
                            train_target = target[c*10 + k:(c+1)*10]
                            test_target = target[c*10 : c*10 + k]
            train_target = np.array(train_target)
            pred_target = []
            for feature in range(5):
                knn = KNN(n_neighbors=2)
                knn.fit(train_data[feature], train_target)
                pred_target.append(knn.predict(test_data[feature]))
            loss = [np.sum(pred_target[feature] != test_target)/len(test_target) for feature in range(5)]
            rez[:, k-1] = loss
            n = 0
            for i in range(2):
                for j in range(3):
                    if i == 1 and j == 2:
                        break
                    axs[i, j].plot(np.arange(1, k+1), rez[n, :k])
                    n+=1
                    fig.show()
                plt.pause(0.05)
        self.iscrossvalidation.setChecked(True)
