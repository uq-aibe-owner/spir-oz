class CustomDict(dict):
    def __init__(self, dic):
        self.dic = dic

    def __getitem__(self, items):
        values = []
        for item in items:
            values.append(self.dic[item])
        return values if len(values) > 1 else values[0]
