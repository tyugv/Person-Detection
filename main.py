import sys
from PyQt5.QtWidgets import QApplication
from windows import Main


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    sys.exit(app.exec_())