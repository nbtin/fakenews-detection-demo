class Function:
    _instance = None
    kind = None

    def __new__(cls, kind=None):
        if cls._instance is None:
            cls._instance = super(Function, cls).__new__(cls)
            cls.kind = kind
        return cls._instance

    def is_available(self):
        if self.kind == 0:
            return False
        elif self.kind == 1:
            return True
        else:
            return False