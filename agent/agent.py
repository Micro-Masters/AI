import tensorflow as tf
import numpy as np
from collections import defaultdict
from tensorflow.contrib import layers


from agent.model import FullyConv


class A2CAgent:
    def __init__(
            self, sess, agent_modifier, use_lstm=False, discount=.99, value_loss_coeff=.5,
            entropy_loss_coeff=.001, learning_rate=1e-4, max_gradient_norm=1.0):
        self.sess = sess
        self.use_lstm = use_lstm
        self.agent_modifier = agent_modifier
        self.discount = discount #Could use discount from sc2_env observation instead, won't need this class var then
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self._build()
        # self._build_ppo()

    # Build and initialize the TensorFlow graph
    def _build(self):
        self.policy, self.action, self.value = self._build_model()
        self.train_operation = self._build_optimizer()
        self.sess.run(tf.global_variables_initializer())

    def _build_ppo(self):
        self.policy, self.action, self.value = self._build_model()
        self.train_operation = self._build_optimizer()

    # Build TensorFlow action (single action from policy) & value operations
    def _build_model(self):
        # Declare model inputs (of any number of observations) and initialize model
        observation_shapes = self.agent_modifier.observation_shapes
        print(observation_shapes['screen'])
        for k, v in observation_shapes.items():
            observation_shapes[k].insert(0, None)

        # for k, v in observation_shapes.items():
        #     observation_shapes[k].insert(0, 1)
        print("HERE ARE TENSOR SHAPES")
        print(observation_shapes['screen'])
        print(observation_shapes['minimap'])
        print(observation_shapes['nonspatial'])
        print(observation_shapes['available_mask'])
        # self.screen_input = tf.placeholder(tf.float32, observation_shapes['screen'], 'screen_input')
        # self.minimap_input = tf.placeholder(tf.float32, observation_shapes['minimap'], 'minimap_input')
        # self.nonspatial_input = tf.placeholder(tf.float32, observation_shapes['nonspatial'], 'nonspatial_input')
        # self.available_mask_input = tf.placeholder(tf.float32, observation_shapes['available_mask'], 'available_mask_input')
        self.screen_input = tf.placeholder(tf.float32, observation_shapes['screen'], 'screen_input')
        self.minimap_input = tf.placeholder(tf.float32, observation_shapes['minimap'], 'minimap_input')
        self.nonspatial_input = tf.placeholder(tf.float32, observation_shapes['nonspatial'], 'nonspatial_input')
        self.available_mask_input = tf.placeholder(tf.float32, observation_shapes['available_mask'], 'available_mask_input')


        self.model = FullyConv(self.agent_modifier.num_actions, self.use_lstm,
                               self.agent_modifier.observation_data_format)

        # Create final action and value operations using model
        policy, value = self.model.build(self.screen_input, self.minimap_input, self.nonspatial_input,
                                         self.available_mask_input, *self.agent_modifier.feature_names)
        action = self.model.sample_action(policy)

        print("policy ")
        print(policy)
        print("action ")
        print(action)
        print("value ")
        print(value)
        print("heh")
        return policy, action, value

    # Build the TensorFlow loss operation
    def _build_optimizer(self):
        # TensorFlow placeholders and compute advantages
        fns = tf.placeholder(tf.int32, [None], 'fns')
        args = {
            k: tf.placeholder(tf.int32, [None], 'args')
            for k in self.policy[1].keys()}
        self.actions = (fns, args)
        self.returns = tf.placeholder(tf.float32, [None], 'returns')
        advantages = tf.stop_gradient(self.returns - self.value)

        # Create loss TensorFlow operation using placeholders
        negative_log_policy = self.model.get_neg_log_prob(self.actions, self.policy)
        policy_loss = tf.reduce_mean(advantages * negative_log_policy)
        value_loss = self.value_loss_coeff * tf.reduce_mean(tf.square(self.value - self.returns))
        entropy_loss = self.entropy_loss_coeff * tf.reduce_mean(self.model.get_entropy(self.policy))

        # Create the final optimizer using loss
        loss = policy_loss + value_loss - entropy_loss
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.99, epsilon=1e-5)
        return layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=None,
            optimizer=optimizer,
            clip_gradients=self.max_gradient_norm,
            name="train_operation")

    # modified from baseline ppo2.py
    def _build_ppo_optimizer(self):

        # Tensorflow placeholders
        returns = tf.placeholder(tf.float32, [None], 'returns')
        values = tf.placeholder(tf.float32, [None], 'values')
        advantages = tf.stop_gradient(returns - values)

        # Create loss TensorFlow operation using placeholders
        R = tf.placeholder(tf.float32, [None])

        clip_range = tf.placeholder(tf.float32, [])
        old_neg_log_policy = tf.placeholdeer(tf.float32, [None])
        negative_log_policy = self.model.get_neg_log_prob(self.actions, self.policy)
        ratio = tf.exp(old_neg_log_policy - negative_log_policy)

        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = tf.reduce_mean(tf.maximum(policy_loss1, policy_loss2))

        old_value_prev = tf.placeholder(tf.float32, [None])
        value_prev = self.value
        value_prev_clipped = old_value_prev + tf.clip_by_value(value_prev - old_value_prev,
                                                   - clip_range, clip_range)
        value_loss1 = tf.square(value_prev - R)
        value_loss2 = tf.square(value_prev_clipped - R)
        value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss1, value_loss2))

        entropy_loss = tf.reduce_mean(self.model.get_entropy(self.policy))

        loss = policy_loss - entropy_loss * entropy_loss_coeff + value_loss * value_loss_coeff

        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)

        return tf.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=lr,
            optimizer=optimizer,
            clip_gradients=self.max_gradient_norm,
            name="train_ppo_operation")

    # Convert policy action syntax to a PySC2 FunctionCall action
    def convert_action(self, policy_action):
        return self.agent_modifier.make_action(policy_action)

    # Take an observation (n_envs length array) and return the action, value
    def act(self, observation):
        feed_dict = self._get_observation_feed(observation)
        print(type(self.action))
        print(type(self.value))
        print(self.action)
        #print("tf print:")
        #print self.value.eval(session=self.sess)
        #tf.Print(self.value)
        print("FEED DICT:")
        # for i in feed_dict:
        # #     print(type(i))
        # print(feed_dict["screen_input"])
        # print(feed_dict["minimap_input"])
        # print(feed_dict["nonspatial_input"])
        # print(feed_dict["available_mask_input"])
        print(type(feed_dict))
        print(type(observation))

        # print(feed_dict[0])
        # print(feed_dict["minimap_input:0"])
        # print(feed_dict["nonspatial_input:0"])
        # print(feed_dict["available_mask_input:0"])
        action, value = self.sess.run([self.action, self.value], feed_dict=feed_dict)
        return action, value

    # Train the model to minimize loss
    def train(self, observations, actions, rewards, dones, values, next_value):
        observations_feed = self._get_observation_feed(observations, train=True)
        actions_feed = self._get_action_feed(actions)
        feed_dict = {
            self.returns: self._get_returns(rewards, dones, values, next_value).flatten(),
            **observations_feed,
            **actions_feed
        }
        self.sess.run(self.train_operation, feed_dict=feed_dict)

    # Compute return values as return = reward + discount * next value * (0 if done else 1)
    def _get_returns(self, rewards, dones, values, next_value):
        returns = np.zeros(rewards.shape, dtype=np.float32)
        # Compute the last return using next_value
        returns[-1] = rewards[-1] + self.discount * next_value * (1 - dones[-1])
        # Compute the rest in a loop using values, working backward
        for i in reversed(range(returns.size[0] - 1)):
            returns[i] = rewards[i] + self.discount * values[i + 1] * (1 - dones[i])
        return returns

    # Modify PySC2 observations and then return the observation input feed
    def _get_observation_feed(self, observation, train=False):
        # Convert PySC2 observations to an array of dictionaries
        if train:
            new_observations = [self.agent_modifier.modify_observations(obs) for obs in observation]
            new_observation = new_observations.reshape(-1, *new_observations.shape[1:])
        else:
            new_observation = self.agent_modifier.modify_observation(observation)

        print("new obs:")
        print(new_observation[0])

        ##TODO: I think the key/value combinations are getting messed up through this process:
        # Convert array of dictionaries to array of arrays
        #observation_input = []

        screen_in = []
        minimap_in = []
        nonspatial_in = []
        masks = []

        #for obs_dict in new_observation:
            # for k, v in obs_dict.items():
        for k, v in new_observation[0].items():
            if k is "screen_features":
                # screen_in = np.array(v)
                screen_in.append(np.array(v))
            elif k is "minimap_features":
               minimap_in.append(np.array(v))
                # minimap_in = np.array(v)
            elif k is "nonspatial_features":
                nonspatial_in.append(np.array(v))
                # nonspatial_in = (np.array(v))
            elif k is "available_mask":
                masks.append(np.array(v))
                # masks = np.array(v)
            else:
                print("OOPS")
            #
            # obs = [np.array(v) for k, v in obs_dict.items()]
            # observation_input.append(obs)

        # screen_in = np.array(screen_in)
        # minimap_in = np.array(minimap_in)
        # nonspatial_in = np.array(nonspatial_in)
        # masks = np.array(masks)

        print("CHECK SHAPES")
        print(np.array(screen_in).shape)

        # observation_input = np.array(observation_input)
        #observation_input = np.array([[screen_in], [minimap_in], [nonspatial_in], [masks]])
        pre_observation_input = np.array([screen_in, minimap_in, nonspatial_in, masks])
        #TODO: will below only work for one game instance?)
        #observation_input = np.array([screen_in[0], minimap_in[0], nonspatial_in[0], masks[0]])
        # print(observation_input.shape)
        # # observation_input = np.transpose(observation_input)
        # # print(len(observation_input[0]))
        #
        # for i in range(4):
        #     observation_input[i] = np.array([pre_observation_input[i][0]])

        #
        # print("0: ", observation_input[0].shape)
        # print("1: ", observation_input[1].shape)
        # print("2: ", observation_input[2].shape)
        # print("3: ", observation_input[3].shape)
        # print(observation_input)
        #
        # #
        # # for i in range(4):
        # #     for j in range(len(observation_input[i])):
        # #         if (observation_input[i][j].shape)[0] is not 1:
        # #             observation_input[i][j] = np.expand_dims(observation_input[i][j], 0)
        #
        # print("SHAPES")
        # print("0: ", observation_input[0][0].shape)
        # print("1: ", observation_input[1][0].shape)
        # print("2: ", observation_input[2][0].shape)
        # print("3: ", observation_input[3][0].shape)

        #
        # for i in range(len(observation_input[0])):
        #     print(observation_input[0][i])
        # batch_xs = np.vstack([np.expand_dims(x, 0) for x in batch_xs])
        # print batch_xs.shape
        #
        # for i in range(4):
        #     observation_input[i] = np.array([pre_observation_input[i][0]])
        return {
            self.screen_input: np.array([pre_observation_input[0][0]]),
            self.minimap_input: np.array([pre_observation_input[1][0]]),
            self.nonspatial_input: np.array([pre_observation_input[2][0]]),
            self.available_mask_input: np.array([pre_observation_input[3][0]]) #XXX
        }

        # return {
        #     self.screen_input: observation_input[0],
        #     self.minimap_input: observation_input[1],
        #     self.nonspatial_input: observation_input[2],
        #     self.available_mask_input: observation_input[3]
        # }

    # Join actions into single fn list and single arg map
    def _get_action_feed(self, actions):
        fns = []
        args = defaultdict(list)

        for fn, arg in actions:
            fns.append(fn)
            for k, v in arg:
                args[k].append(v)
        return {self.actions: (np.array(fns).flatten(), args)}