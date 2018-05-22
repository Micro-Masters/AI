import tensorflow as tf

class A2CAgent:
    def __init__(self, sess, policy, learning_rate=1e-4, max_gradient_norm=1):
        print('A2C init')
        self.sess = sess
        self.policy = policy
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm

    def _build(self):
        self.action, self.value = self.policy.build()
        #loss =

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.99, epsilon=1e-5)
        self.train_op = tf.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=optimizer,
            clip_gradients=self.max_gradient_norm,
            learning_rate=None,
            name="train_op")

    def train(self):
        print('A2C train')

    def act(self, observation):
        #pass observation
        return self.sess.run([self.action, self.value])