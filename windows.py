from PyQt5.QtWidgets import QCheckBox, QWidget, QTableWidget, QTableWidgetItem, QLabel, QPushButton
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neighbors import KNeighborsClassifier as KNN
from statistics import mode
from feature_selection import *
from graphics import show_features, show_features_progress


class Main(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.resize(430, 400)
        self.move(900, 200)

        self.images = None
        self.target = None
        self.classes = None
        self.features_data = None

        self.isdatasetloaded = QCheckBox("Загрузить датасет", self)
        self.isdatasetloaded.move(10, 10)

        self.isfeaturesgot = QCheckBox("Получение признаков", self)
        self.isfeaturesgot.move(10, 30)

        self.iscrossvalidation = QCheckBox("Кросс-Валидация", self)
        self.iscrossvalidation.move(10, 50)

        self.state = 0

        self.process_button = QPushButton(self)
        self.process_button.setText("Начать работу")
        self.process_button.setGeometry(10, 90, 150, 30)
        self.process_button.clicked.connect(self.work_process)

        self.table_label = QLabel(self)
        self.table_label.setText("Лучшие результаты классификации на тестовой выборке")
        self.table_label.setGeometry(50, 130, 300, 20)

        self.table = QTableWidget(7, 4, self)
        self.table.setGeometry(10, 150, 400, 300)
        feature_names = ['Гистограмма', 'DFT', 'DCT', 'Scale', 'Градиент', 'Большинство']
        column_names = ['Признак', 'Правильно', 'Неправильно', 'Точность']
        for c in range(4):
            item = QTableWidgetItem(column_names[c])
            self.table.setItem(0, c, item)
        for r in range(6):
            item = QTableWidgetItem(feature_names[r])
            self.table.setItem(r + 1, 0, item)

        self.show()

    def work_process(self):

        def get_by_max_choice(pred_target):
            preds = np.array(pred_target)
            result_pred = []
            for i in range(len(preds[0])):
                result_pred.append(mode(preds[:, i]))
            return result_pred

        if self.state == 0:
            data_images = fetch_olivetti_faces()
            self.images = data_images['images']
            self.target = data_images['target']
            self.classes = np.unique(self.target)[-1]
            self.process_button.setText("Получить признаки")
            self.isdatasetloaded.setChecked(True)
            self.state += 1

        elif self.state == 1:
            features_data = []
            functions = [histogram, dft, dct, mean_pooling, gradient]
            features_lens = [len(histogram(self.images[0])), len(dft(self.images[0]).flatten()), len(dct(self.images[0]).flatten()),
                                len(mean_pooling(self.images[0]).flatten()), len(gradient(self.images[0]))]
            for f, n in zip(functions, features_lens):
                features_data.append(make_dataset(self.images, f, n))
            self.features_data = features_data
            self.process_button.setText("Кросс-Валидация")
            self.isfeaturesgot.setChecked(True)
            show_features(self.images[0])
            self.state += 1

        elif self.state == 2:
            fig, axs = show_features_progress()
            rez = np.zeros((6, 9))     
            for k in range(1, 10):
                train_data = []
                train_target = []
                test_data = []
                test_target = []

                for feature in range(5):
                    train_data.append([])
                    test_data.append([])
                    for c in range(self.classes):
                        if len(train_data[feature]) > 0:
                            test_data[feature] = np.concatenate((test_data[feature], self.features_data[feature][c*10 + k:(c+1)*10]))
                            train_data[feature] = np.concatenate((train_data[feature], self.features_data[feature][c*10 : c*10 + k]))
                            if feature == 0:
                                test_target = np.concatenate((test_target, self.target[c*10 + k:(c+1)*10]))
                                train_target = np.concatenate((train_target, self.target[c*10 : c*10 + k]))
                        else:
                            for arr in self.features_data[feature][c*10 + k:(c+1)*10]:
                                test_data[feature].append(arr)
                            for arr in self.features_data[feature][c*10 : c*10 + k]:
                                train_data[feature].append(arr)
                            if feature == 0:
                                test_target = self.target[c*10 + k:(c+1)*10]
                                train_target = self.target[c*10 : c*10 + k]
                train_target = np.array(train_target)
                test_target = np.array(test_target)
                pred_target = []
                for feature in range(5):
                    knn = KNN(n_neighbors=2)
                    knn.fit(train_data[feature], train_target)
                    pred_target.append(knn.predict(test_data[feature]))
                pred_target.append(get_by_max_choice(pred_target))
                acc = [np.sum(pred_target[feature] == test_target)/len(test_target) for feature in range(6)]
                rez[:, k-1] = acc
                n = 0
                for i in range(2):
                    for j in range(3):
                        axs[i, j].plot(np.arange(1, k+1) * self.classes, rez[n, :k])
                        n+=1
                        fig.show()
                    plt.pause(0.01)

            self.pred_target = pred_target
            self.test_target = test_target
            self.process_button.setText("Получить результаты")
            self.iscrossvalidation.setChecked(True)
            self.state += 1

        elif self.state == 3:
            for feature in range(6):
                right_ans = np.sum(self.pred_target[feature] == self.test_target)
                self.table.setItem(feature + 1, 1, QTableWidgetItem(str(right_ans)))
                item = QTableWidgetItem(str(len(self.test_target) - right_ans))
                self.table.setItem(feature + 1, 2, item)
                item = QTableWidgetItem(str(round(right_ans / len(self.test_target), 3)))
                self.table.setItem(feature + 1, 3, item)
            self.process_button.hide()
            self.state += 1

        else:
            return
