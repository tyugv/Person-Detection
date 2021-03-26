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

        self.iscrossvalidation = QCheckBox("Кросс-Валидация", self)
        self.iscrossvalidation.move(10, 30)

        self.iscrossvalidation = QCheckBox("Получение признаков", self)
        self.iscrossvalidation.move(10, 50)

        self.show()
        self.work_process()

    def work_process(self):
        data_images = fetch_olivetti_faces()
        images = data_images['images']
        target = data_images['target']
        classes = np.unique(target)[-1]
        self.isdatasetloaded.setChecked(True)
        show_features(images[0])
        functions = [histogram, dft, dct, mean_pooling, gradient]

        fig, axs = show_features_progress()
        rez = []
        for k in range(10):
            rez.append(k/10)
            for i in range(2):
                for j in range(3):
                    axs[i, j].plot(rez.pop())
                    fig.show()
            plt.pause(0.05)
        #fig.show()
        print('11')
