import os

class FileClipboard:
    clipboard_file = '/tmp/clipboard.txt'

    @staticmethod
    def copy(text):
        with open(FileClipboard.clipboard_file, 'w') as file:
            file.write(text)

    @staticmethod
    def paste():
        if os.path.exists(FileClipboard.clipboard_file):
            with open(FileClipboard.clipboard_file, 'r') as file:
                return file.read()
        return ""

class CustomClipboard:
    def __init__(self):
        self._copy = FileClipboard.copy
        self._paste = FileClipboard.paste

    def copy(self, text):
        self._copy(text)

    def paste(self):
        return self._paste()

