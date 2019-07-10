import chainer
import chainer.links as L
import chainer.functions as F

class DQN(chainer.Chain):
    def __init__(self, n_actions):
        super(DQN, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=None, out_channels=32, ksize=8, stride=4)
            self.conv2 = L.Convolution2D(in_channels=32, out_channels=64, ksize=4, stride=2)
            self.conv3 = L.Convolution2D(in_channels=64, out_channels=64, ksize=3, stride=1)

            self.fc1 = L.Linear(None, 512)
            self.fc2 = L.Linear(512, n_actions)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc1(h.reshape(h.shape[0], -1)))  # not sure if this line works for chainer
        h = self.fc2(h)
        return h
