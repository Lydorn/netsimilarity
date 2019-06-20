
class AccumulatedAccuracyMetric:

    def __init__(self):
        self.reset()

    def __call__(self, outputs, targets):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        cur_correct = pred.eq(targets[0].data.view_as(pred)).cpu().sum()
        cur_total = targets[0].size(0)
        self.correct += cur_correct
        self.total += cur_total
        return 100 * float(cur_correct) / cur_total

    def reset(self):
        self.correct = 0
        self.total = 0

    def get_accumulated_value(self):
        return 100 * float(self.correct) / self.total
