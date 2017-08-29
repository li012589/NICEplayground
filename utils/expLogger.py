import math

class expLogger:
    def __init__(self,d):
        self.dic = d
    def cal(self,delta):
        if delta in self.dic:
            return self.dict[delta]
        else:
            self.dic[delta] = math.exp(-delta)
            return self.dic[delta]