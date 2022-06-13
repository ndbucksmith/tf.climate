from operator import itemgetter

import tensorflow as tf
import wc_batcher as wcb
import numpy as np
import time
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Input
from keras.models import Model

MODEL_TYPE = "LSTM"
PRINT_12MO_STATS = True


class KerasSeqLSTM():
    def __init__(self, params):
        """
        Sequential LSTM climate model for 12 month average temps as function of radiation,
        precipitation, altitude, albedo, hemisphere
        Best accuracy so far is 3 to 5 C mean squared error
        :param params:
        """

        cell_size = params["cell_size"]
        self.cell_size = cell_size
        xin_size = params["rxin_size"]
        self.xin_size = xin_size
        b_size = params["batch_size"]
        y_size = 12
        self.params = params
        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
        input_width = len(params['take']) + len(params['rnn_take'])
        # Add a LSTM layer with 128 internal units.
        # self.model.add(tf.keras.layers.LSTM(128))
        l1 = tf.keras.layers.Dense(16, name="monthly_inputs", batch_input_shape=(120, 12, input_width))
        self.model.add(l1)
        # l2 = tf.keras.layers.Dense(20, name="monthly_expanded")
        # self.model.add(l2)
        self.model.add(
            tf.keras.layers.LSTM(cell_size, go_backwards=False, stateful=True, batch_size=(120, 12, input_width)))
        self.model.add(tf.keras.layers.Dense(12))
        # self.model.add_loss(tf.keras.losses.MeanSquaredError())
        self.model.build((None, 12, xin_size))
        self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError)
        for wt in self.model.weights:
            print("weight shape:", wt.shape)
        self.model.compute_loss = self.c_l
        self.model.optimizer.learning_rate = 0.1
        print(self.model.summary())

    def c_l(self, x=None, y=None, y_pred=None, sample_weight=None):
        h_ = self.model(x)
        self.losses = tf.square(tf.subtract(h_, y))
        loss = tf.reduce_mean(self.losses)
        # print("loss computed:", loss)
        return loss


class Keras3moModel():
    def __init__(self, params):
        cell_size = params["cell_size"]
        self.cell_size = cell_size
        xin_size = 3 + 3 + 1
        self.xin_size = xin_size
        b_size = params["batch_size"]
        y_size = 12
        self.params = params
        self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))

        # Add a LSTM layer with 128 internal units.
        # self.model.add(tf.keras.layers.LSTM(128))
        l1 = tf.keras.layers.Dense(xin_size, name="2by3powersplusaltitude")
        self.model.add(l1)
        l2 = tf.keras.layers.Dense(20, name="ins_expanded")
        self.model.add(l2)
        self.model.add(tf.keras.layers.Dense(1))
        # self.model.add_loss(tf.keras.losses.MeanSquaredError())
        self.model.build((None, xin_size))
        self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError)
        for wt in self.model.weights:
            print("weight shape:", wt.shape)
        self.model.compute_loss = self.c_l
        self.model.optimizer.learning_rate = 0.1
        print(self.model.optimizer.learning_rate)

    def c_l(self, x=None, y=None, y_pred=None, sample_weight=None):
        h_ = self.model(x)
        self.losses = tf.square(tf.subtract(h_, y))
        loss = tf.reduce_mean(self.losses)
        # print("loss computed:", loss)
        return loss


def simple_train(model, x, y):
    # Run forward pass.
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = model.compute_loss(x, y, y_pred)
    # self._validate_target_and_loss(y, loss)
    # Run backwards pass.
    model.optimizer.minimize(loss, model.trainable_variables, tape=tape)
    return model.compute_metrics(x, y, y_pred, None)


params = {}
batch_size = 120
params["cell_size"] = 64
params["rxin_size"] = 8
params["batch_size"] = batch_size
params["take"] = [
    1,
    3, 8, 9, 10, 11, 12,
    13,
    14,
    15,
]
params["rnn_take"] = [
    0,
    1,
    2,
    3,
]

if MODEL_TYPE == 'LSTM':
    h = KerasSeqLSTM(params)  # type KerasModel
else:
    h = Keras3moModel(params)
print("list of weights")
for wt in h.model.weights:
    print(wt.shape)
# self.model.compute_loss = self.c_l
h.model.optimizer.learning_rate = 0.01
print("learning rate:", h.model.optimizer.learning_rate)

zb = wcb.zbatch(batch_size)
start_time = time.time()
input_width = len(params['take']) + len(params['rnn_take'])
for _bx in range(500000):
    #  np.array(ins_bat), rnn_seqs, wc_trs, rnn_trus, d3_idx

    tb = zb.zbatch(batch_size, True)
    static_ins = np.take(tb[0], params["take"], axis=1)

    rnn_ins = np.take(tb[1], [0, 1, 2, 3], axis=2)
    # print("rnn ins shape ", rnn_ins.shape)
    if MODEL_TYPE == 'LSTM':
        sfb = wcb.small_flat_batch(static_ins=static_ins, rnn_ins=rnn_ins)
        # lat_max = -20.0
        # for ix in range(120):
        #     for mx in range(12):
        #         if sfb[ix][mx][4] > lat_max:
        #             lat_max = sfb[ix][mx][4]
        # print(lat_max)
        # print("batch trues shape ", len(tb[3]), tb[3][0].shape)
        trues = np.reshape(tb[3], [batch_size, 12])
        x = np.array(sfb)
        # for ix in range(120):
        #     for mx in range(12):
        #         if x[ix, mx, 4] > lat_max:
        #             lat_max = x[ix, mx, 4]
        # print(lat_max)
    else:
        sfb_3mo = []
        trues = []
        for bx in range(batch_size):
            for mx in range(12):
                sfb_3mo.append([rnn_ins[bx, 0, mx], rnn_ins[bx, 0, mx - 1], rnn_ins[bx, 0, mx - 2],
                                rnn_ins[bx, 2, mx], rnn_ins[bx, 2, mx - 1], rnn_ins[bx, 2, mx - 2],
                                static_ins[bx, 0]])
                trues.append(tb[3][bx][mx])
        x = np.array(sfb_3mo)
        trues = np.array(trues)
    train_res = simple_train(h.model, x, trues)
    # print("train ", h.model.train_step)
    y_ = h.model(x)
    if _bx % 100 == 0 and PRINT_12MO_STATS:
        print(f"training iteration {_bx}")
        print(f"first weights {h.model.weights[0][0][0]} {h.model.weights[1][0]}")
        np_loss = h.losses.numpy()  # type np.ndarray
        print(f"loss {np_loss.mean()}   median {np.median(np_loss)}")
        print(f"first points {y_[0, 0].numpy()}", trues[0, 0])
        l_max = -1
        i_max = -1
        j_max = -1
        lat_at_lmax = -10.0
        greater_than_2ct = 0
        greater_than_4ct = 0
        neg_losses = [];
        pos_losses = []
        for ix in range(len(trues)):
            for jx in range(len(trues[0])):
                raw_loss = y_[ix, jx].numpy() - trues[ix, jx]
                if raw_loss >= 9.0:
                    pos_losses.append([raw_loss, x[ix, jx, 4]])
                elif raw_loss <= -9.0:
                    neg_losses.append([raw_loss, x[ix, jx, 4]])
                if l_max < abs(y_[ix, jx].numpy() - trues[ix, jx]):
                    l_max = abs(y_[ix, jx].numpy() - trues[ix, jx])
                    i_max = ix
                    j_max = jx
                    lat_at_lmax = x[ix, jx, 4]
                if abs(y_[ix, jx].numpy() - trues[ix, jx]) > 2.0:
                    greater_than_2ct += 1
                if abs(y_[ix, jx].numpy() - trues[ix, jx]) > 4.0:
                    greater_than_4ct += 1
        print(f"max loss {l_max} at batch {i_max} at month {j_max} at lattitude {lat_at_lmax}")
        if len(pos_losses) < 6:
            print("big + errors", pos_losses)
        if len(neg_losses) < 6:
            print("big - errors", neg_losses)
        # pos_losses.sort(key=itemgetter(0))
        # neg_losses.sort(key=itemgetter(0))
        # print(f"max + loss {pos_losses[-1]} {pos_losses[-2]} {pos_losses[-3]}")
        # print(f"max - loss {neg_losses[0]} {neg_losses[1]} {neg_losses[2]}")
        print(f"months with error > 2C {greater_than_2ct}")
        print(f"months with error > 4C {greater_than_4ct}")

print(f"training time {time.time() - start_time} seconds for {_bx} train steps")
