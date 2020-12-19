import numpy as np
from os.path import join
from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger


class Viterbi:
    def __init__(self, grammar, probs, transition=0.5, framesampling=1):
        self._grammar = grammar
        self._framesampling = framesampling
        self._transition_self = -np.log(transition)
        self._transition_next = -np.log(1 - transition)
        self._transitions = np.array([self._transition_self, self._transition_next])
        # self._transitions = np.array([0, 0])

        self._state = []

        self._probs = probs
        self._state = self._probs[0, 0]
        self._number_frames = self._probs.shape[0]
        self._n_segs = int(np.ceil(self._number_frames/self._framesampling))

        # probabilities matrix
        self._T1 = np.zeros((len(self._grammar), self._n_segs)) + np.inf
        start_state = self._grammar.states()[0]
        self._seg_idx = 0
        self._T1[start_state, 0] = self.get_prob(start_state)
        # argmax matrix
        self._T2 = np.zeros((len(self._grammar), self._n_segs)) + np.inf
        self._T2[start_state, 0] = 0

        self._frame_idx = 1
        self._seg_idx = 1

    # @timing
    def inference(self):
        while self._seg_idx < self._n_segs:
            for state_idx, state in enumerate(self._grammar.states()):
                idxs = np.array([max(state_idx - 1, 0), state_idx])
                probs = self._T1[idxs, self._seg_idx - 1] + \
                        self._transitions[idxs - max(state_idx - 1, 0)] + \
                        self.get_prob(state)
                self._T1[state_idx, self._seg_idx] = np.min(probs)
                self._T2[state_idx, self._seg_idx] = np.argmin(probs) + \
                                                       max(state_idx - 1, 0)
            self._seg_idx += 1

    def get_prob(self, state):
        start_idx = self._seg_idx * self._framesampling
        end_idx = min((self._seg_idx + 1) * self._framesampling, self._number_frames)
        return np.sum(self._probs[start_idx:end_idx, state], axis=0)

    # # @timing
    # def backward(self, strict=True):
    #     if strict:
    #         last_state = -1 if self._T2.shape[0] < self._T2.shape[1] else self._T2.shape[1]
    #         self._grammar.set_framewise_state(self._T2[last_state, -1], last=True)
    #     else:
    #         self._grammar.set_framewise_state(self._T1[..., -1], last=True)

    #     for i in range(self._T1.shape[1] - 1, 0, -1):
    #         self._grammar.set_framewise_state(self._T2[..., i])
    #     self._grammar.reverse()

    # @timing
    def backward(self, strict=True):
        last_iter = self._number_frames - (self._n_segs - 1) * self._framesampling
        if strict:
            last_state = -1 if self._T2.shape[0] < self._T2.shape[1] else self._T2.shape[1]
            self._grammar.set_framewise_state(self._T2[last_state, -1], last_iter, last=True)
        else:
            self._grammar.set_framewise_state(self._T1[..., -1], last_iter, last=True)

        for i in range(self._T1.shape[1] - 1, 0, -1):
            self._grammar.set_framewise_state(self._T2[..., i], self._framesampling)
        self._grammar.reverse()

    def save_framewise_states(self):
        np.savetxt(join(opt.root, opt.save, self._filename + '_' + self._grammar.name + '.txt'),
                   self._grammar.framewise_states(), fmt='%d')

    def loglikelyhood(self):
        return -1 * self._T1[-1, -1]

    def alignment(self):
        return self._grammar.framewise_states()

    def calc(self, alignment):
        self._sum = np.sum(np.abs(self._probs[np.arange(self._number_frames), alignment]))
