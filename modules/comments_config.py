class Configuration:
    def __init__(self):
        self._enable_word2vec = False
        self._enable_elmo = False
        self._enable_roberta = False
        self._enable_use = True

        self._threshold_word2vec = 0.97
        self._threshold_elmo = 0.99
        self._threshold_roberta = 0.90
        self._threshold_use = 0.90

    def enable_word2vec(self):
        return self._enable_word2vec

    def set_enable_word2vec(self, value):
        self._enable_word2vec = value

    def enable_elmo(self):
        return self._enable_elmo

    def set_enable_elmo(self, value):
        self._enable_elmo = value

    def enable_roberta(self):
        return self._enable_roberta

    def set_enable_roberta(self, value):
        self._enable_roberta = value

    def enable_use(self):
        return self._enable_use

    def set_enable_use(self, value):
        self._enable_use = value

    def threshold_word2vec(self):
        return self._threshold_word2vec

    def set_threshold_word2vec(self, value):
        self._threshold_word2vec = value

    def threshold_elmo(self):
        return self._threshold_elmo

    def set_threshold_elmo(self, value):
        self._threshold_elmo = value

    def threshold_roberta(self):
        return self._threshold_roberta

    def set_threshold_roberta(self, value):
        self._threshold_roberta = value

    def threshold_use(self):
        return self._threshold_use

    def set_threshold_use(self, value):
        self._threshold_use = value

config = Configuration()
