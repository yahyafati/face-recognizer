class Face:
    
    def __init__(self, name, encoding, location = None, best_match_index=None):
        self.name = name
        self.encoding = encoding
        self.location = location
        self.best_match_index = best_match_index
        self.confidence = None
    
    def __str__(self):
        if self.name is None:
            return "Unknown"
        return f"{self.name} {self.confidence}"
    
    def __repr__(self):
        return f"Face({self.name}, {self.location}, {self.best_match_index}, {self.confidence})"