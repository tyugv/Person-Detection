from PyQt5.QtWidgets import QCheckBox, QWidget
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split


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

        train_idx, test_idx = train_test_split(np.arange(len(images)), test_size=0.25)
        train_images, train_target = images[train_idx], target[train_idx]
        test_images, test_target = images[test_idx], target[test_idx]

