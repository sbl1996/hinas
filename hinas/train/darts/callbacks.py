from horch.train.callbacks import Callback


class PrintGenotype(Callback):

    def __init__(self, after_epochs):
        super().__init__()
        self.after_epochs = after_epochs

    def after_epoch(self, state):
        if state['epoch'] < self.after_epochs:
            return
        p = """Genotype(
    normal=[
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
    ], normal_concat=[2, 3, 4, 5],
    reduce=[
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
    ], reduce_concat=[2, 3, 4, 5],
)"""
        g = self.learner.model.genotype()
        print(p % (tuple(g.normal) + tuple(g.reduce)))


class TrainArchSchedule(Callback):

    def __init__(self, after_epochs):
        super().__init__()
        self.after_epochs = after_epochs

    def begin_epoch(self, state):
        self.learner.train_arch = state['epoch'] >= self.after_epochs


class TauSchedule(Callback):

    def __init__(self, tau_max, tau_min):
        super().__init__()
        self.tau_max, self.tau_min = tau_max, tau_min

    def begin_epoch(self, state):
        tau_max = self.tau_max
        tau_min = self.tau_min
        tau = tau_max - (tau_max - tau_min) * (state['epoch'] / state['epochs'])
        self.learner.model.tau = tau