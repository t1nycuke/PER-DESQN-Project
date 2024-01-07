import numpy as np
import tensorflow as tf

def correct_dimensions(s, targetlength):

    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s


class ESN():

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001, lr = 0.01, teacher_forcing=True,
                 input_shift=None, input_scaling=None,
                 teacher_scaling=None, teacher_shift=None,
                 out_activation=lambda x: x, inverse_out_activation=lambda x: x,
                 random_state=None):

        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise

        self.input_shift = correct_dimensions(input_shift, n_inputs)
        self.input_scaling = correct_dimensions(input_scaling, n_inputs)

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state
        
        self.lr = lr


        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.initweights()

        self.laststate = np.zeros(self.n_reservoir)
        self.lastinput = np.zeros(self.n_inputs)
        self.lastoutput = np.zeros(self.n_outputs)

        self.startstate = np.zeros(self.n_reservoir)
        self.startinput = np.zeros(self.n_inputs)
        self.startoutput = np.zeros(self.n_outputs)
    def initweights(self):

        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5

        W[self.random_state_.rand(W.shape[0], W.shape[1]) < self.sparsity] = 0

        radius = np.max(np.abs(np.linalg.eigvals(W)))

        self.W = W * (self.spectral_radius / radius)


        self.W_in = self.random_state_.rand(
            self.n_reservoir, self.n_inputs) * 2 - 1

        self.W_feedb = self.random_state_.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1

        self.W_out = self.random_state_.rand(
            self.n_outputs, self.n_reservoir + self.n_inputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):

        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_feedb, output_pattern))
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))
        return (np.tanh(preactivation)
                + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def _scale_inputs(self, inputs):

        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift
        return inputs

    def _scale_teacher(self, teacher):

        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):

        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, inputs, outputs, nForgetPoints):


        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        inputs_scaled = self._scale_inputs(inputs)
        teachers_scaled = self._scale_teacher(outputs)


        states = np.vstack(
            [self.startstate, np.zeros((inputs.shape[0], self.n_reservoir))])
        inputs_scaled = np.vstack(
            [self.startinput, inputs_scaled])
        teachers_scaled = np.vstack(
            [self.startoutput, teachers_scaled])

        for n in range(inputs.shape[0]):
            states[n+1, :] = self._update(states[n], inputs_scaled[n+1, :], teachers_scaled[n, :])
            if ((n+1) > nForgetPoints):
                extended_states = np.hstack((states[n+1, :], inputs_scaled[n+1, :]))
                error = (np.dot(self.W_out, extended_states.T) - teachers_scaled[n+1, :])
                self.W_out = self.W_out - self.lr * np.dot(error.reshape((-1, 1)), extended_states.reshape((1, -1)))



        extended_states = np.hstack((states[1:, :], inputs_scaled[1:, :]))



        pred_train = self._unscale_teacher(self.out_activation(
            np.dot(extended_states, self.W_out.T)))

        return pred_train

    def predict(self, inputs, nForgetPoints, continuation=True):

        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = self.startstate
            lastinput = self.startinput
            lastoutput = self.startoutput

        inputs = np.vstack([lastinput, self._scale_inputs(inputs)])
        states = np.vstack(
            [laststate, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack(
            [lastoutput, np.zeros((n_samples, self.n_outputs))])

        for n in range(n_samples):
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            outputs[n + 1, :] = self.out_activation(np.dot(self.W_out,
                                                           np.concatenate([states[n + 1, :], inputs[n + 1, :]])))

        if continuation:

            self.laststate = states[-1, :]
            self.lastinput = inputs[-1, :]
            self.lastoutput = outputs[-1, :]

        transient = nForgetPoints+1
        return self._unscale_teacher(self.out_activation(outputs[transient:]))