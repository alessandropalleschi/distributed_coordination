import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from memory import PPOMemory
from networks import ActorNetwork, CriticNetwork
from tensorflow.keras.callbacks import TensorBoard
import datetime
    
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64,
                 n_epochs=10, chkpt_dir='models/'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        self.running_stats = []
        self.actor = ActorNetwork(n_actions)
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNetwork()
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size, gamma, gae_lambda)
        self.log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.learning_steps = 0
    
    def normalize_states(self, states):
        normalized_states = np.zeros_like(states)

        for channel, stats in enumerate(self.running_stats):
            channel_data = states[:, channel:channel + 1, :]
            mean, std = stats.get_stats()
            
            # Normalize along the feature dimension
            normalized_channel_data = (channel_data - mean) / std
            normalized_states[:, channel:channel + 1, :] = normalized_channel_data

        return normalized_states
    
    def store_transition(self, state, action, probs, vals, reward, done, agent_id):
        self.memory.store_memory(state, action, probs, vals, reward, done, agent_id)

    def store_last_value(self, value, agent_id):
        self.memory.store_last_value(value, agent_id)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir + 'actor_' + self.date) 
        self.critic.save(self.chkpt_dir + 'critic_' + self.date)

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir + 'actor_20240115-175903')
        self.critic = keras.models.load_model(self.chkpt_dir + 'critic_20240115-175903')
        self.actor.log_std = tf.Variable(-0.5 * tf.ones((1, 2), dtype=tf.float32))

    def compute_value(self, observation):
        state = tf.convert_to_tensor(observation)
        value = self.critic(state)
        value = value.numpy()
        return value
    
    def choose_action(self, observation):
        state = tf.convert_to_tensor(observation)

        mean = self.actor(state)
        log_std_expanded = tf.tile(self.actor.log_std, [tf.shape(mean)[0], 1])
        std = tf.exp(log_std_expanded)

        dist = tfp.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = tf.math.reduce_sum(dist.log_prob(action), axis=-1, keepdims=True)
        
        value = self.critic(state)
        action = action.numpy()
        value = value.numpy()
        log_prob = log_prob.numpy()
        return action, log_prob, value

    def learn(self):

        self.avg_actor_loss = 0
        self.avg_critic_loss = 0
        self.avg_entropy_loss = 0

        for _ in range(self.n_epochs):
            avg_actor_loss_epoch = 0
            avg_critic_loss_epoch = 0
            avg_entropy_loss_epoch = 0
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, advantage, returns, batches = \
                self.memory.generate_batches()
            actors_iters  = 0
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])
                    means = self.actor(states)
                    
                    log_std_expanded = tf.tile(self.actor.log_std, [tf.shape(means)[0], 1])
                    std = tf.math.exp(log_std_expanded)
                    dist = tfp.distributions.Normal(means, std)
                    new_probs = tf.math.reduce_sum(dist.log_prob(actions), axis=-1, keepdims=True)


                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

           # KL Divergence calculation
                    kl_divergence = tf.reduce_mean(self.kl_divergence(old_probs,new_probs))
                    # Early stopping based on KL Divergence
                    entropy = tf.reduce_mean(dist.entropy())
                    # Total actor loss with entropy term
                    actor_loss_plus_entropy = actor_loss + 0.01 * entropy
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(
                    #                                  returns-critic_value, 2))
                    actors_iters +=1
                    
                    avg_actor_loss_epoch += actor_loss
                    avg_entropy_loss_epoch += entropy

                    if kl_divergence > 4*15*1e-4:
                        break

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss_plus_entropy, actor_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                    # Entropy term
            critic_iters  = 0
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr[batch])
                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    critic_loss = 0.5*keras.losses.MSE(critic_value, returns[batch])
                    avg_critic_loss_epoch += critic_loss

                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))
                critic_iters+=1

            avg_entropy_loss_epoch /= actors_iters
            avg_actor_loss_epoch /= actors_iters
            avg_critic_loss_epoch /= critic_iters
            
            self.avg_actor_loss += avg_actor_loss_epoch
            self.avg_critic_loss += avg_critic_loss_epoch
            self.avg_entropy_loss += avg_entropy_loss_epoch
        
        self.avg_actor_loss /= self.n_epochs
        self.avg_critic_loss /= self.n_epochs
        self.avg_entropy_loss /= self.n_epochs

        self.learning_steps +=1
        self.memory.clear_memory()
    
    
    def kl_divergence(self, old_probs, new_probs):
        old_probs = tf.exp(old_probs)
        new_probs = tf.exp(new_probs)
        
        kl_div = tf.reduce_sum(old_probs * (tf.math.log(old_probs) - tf.math.log(new_probs)), axis=-1)
        kl_div = tf.reduce_mean(kl_div)
        
        return kl_div

    def log_statistics(self):
        with tf.summary.create_file_writer(self.log_dir).as_default():
            tf.summary.scalar('Actor Loss', self.avg_actor_loss, step=self.learning_steps)
            tf.summary.scalar('Critic Loss', self.avg_critic_loss, step=self.learning_steps)
            tf.summary.scalar('Entropy Loss', self.avg_entropy_loss, step=self.learning_steps)
            tf.summary.scalar('Return', self.scores, step=self.learning_steps)
            tf.summary.scalar('Average Return', self.avg_scores, step=self.learning_steps)
            tf.summary.scalar('Best Score', self.best_score, step=self.learning_steps)
            tf.summary.scalar('Forward Velocity std', tf.reduce_mean(tf.exp(self.actor.log_std)[:,0]), step=self.learning_steps)
            tf.summary.scalar('Angular Velocity std', tf.reduce_mean(tf.exp(self.actor.log_std)[:,1]), step=self.learning_steps)