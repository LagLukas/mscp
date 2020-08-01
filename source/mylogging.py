import json


class Logging:

    def __init__(self, path):
        self.step = 0
        self.results = []
        self.path = path

    def log_entry(self, iteration, best, time):
        self.results.append([iteration, best, time])

    def save(self):
        with open(self.path, "w") as file:
            json.dump(self.results, file, sort_keys=True, indent=4)
        del self.results