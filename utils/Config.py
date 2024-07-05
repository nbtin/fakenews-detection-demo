class Config:

    def __init__(self, kind, roc_value, roc_service):
        self.kind = kind
        self.roc_value = roc_value
        self.roc_service = roc_service

    def __str__(self):
        return f"kind: {self.kind}, roc_value: {self.roc_value}, roc_service: {self.roc_service}"

    def get_kind(self):
        if self.kind == "Cheapfakes (Ours)":
            return 0
        elif self.kind == "Manipulated Images (TruFor)":
            return 1
        else:
            return 2

    def get_roc_value(self):
        return self.roc_value
    
    def get_roc_service(self):
        return self.roc_service