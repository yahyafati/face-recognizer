import random

class Face:

    @staticmethod
    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    @staticmethod
    def random_safe_color():
        # make sure the color is not too close to red
        color = Face.random_color()
        while color[0] > 200 and color[1] < 100 and color[2] < 100:
            color = Face.random_color()
        return color

    @staticmethod
    def calculate_center(location, scale=1):
        top, right, bottom, left = location
        return ((left + right) // 2) * scale, ((top + bottom) // 2) * scale
    
    def __init__(self, name, encoding, location = None, best_match_index=None):
        self.name = name
        self.encoding = encoding
        self.location = location
        self.best_match_index = best_match_index
        self.confidence = None
        self.previous_locations = []
        self.color = Face.random_safe_color()
        self.score = 0
    
    def get_center(self, scale=1):
        return Face.calculate_center(self.location, scale)
    
    def get_location(self, scale):
        top, right, bottom, left = self.location
        return (top * scale, right * scale, bottom * scale, left * scale)
    
    def is_in_face(self, x, y, scale=1):
        top, right, bottom, left = self.get_location(scale)
        return left <= x <= right and top <= y <= bottom
    
    def get_previous_centers(self, scale=1):
        return [Face.calculate_center(location, scale) for location in self.previous_locations]
    
    def __str__(self):
        if self.name is None:
            return "Unknown"
        return f"{self.name} {self.confidence}"
    
    def __repr__(self):
        return f"Face({self.name}, {self.location}, {self.best_match_index}, {self.confidence})"