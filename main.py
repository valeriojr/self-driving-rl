import collections
import statistics
from datetime import datetime
from typing import Tuple, List

import gymnasium as gym
import numpy
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.losses as losses
import tqdm

import self_driving_car


# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.
def env_step(action: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Returns state, reward and done flag given an action."""
    state, reward, done, truncated, info = env.step(action)
    if truncated:
        reward += env.vehicle.position.z
    return *state, numpy.array(reward, numpy.float32), numpy.array(done, numpy.int32)


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action],
                             [tf.uint8, tf.float32, tf.float32, tf.int32])


def run_episode(initial_state: tf.Tensor, model: tf.keras.Model, max_steps: int) -> Tuple[
    tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    camera, vehicle_state = initial_state
    initial_state_shape = camera.shape

    for t in tf.range(max_steps):
        # state = tf.expand_dims(state, 0)
        action_logits_t, value = model([tf.expand_dims(camera, axis=0), tf.expand_dims(vehicle_state, axis=0)])

        action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        values = values.write(t, tf.squeeze(value))
        action_probs = action_probs.write(t, action_probs_t[0, action])

        camera, vehicle_state, reward, done = tf_env_step(action)
        camera.set_shape(initial_state_shape)
        vehicle_state.set_shape((2,))
        # state.set_shape(initial_state_shape)

        rewards = rewards.write(t, reward)
        env.render()

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    with train_summary_writer.as_default():
        tf.summary.image('Camera', [camera])

    return action_probs, values, rewards


def get_expected_return(rewards: tf.Tensor, gamma: float, standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + epsilon))

    return returns


def compute_loss(action_probs: tf.Tensor, values: tf.Tensor,
                 returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


# @tf.function
def train_step(initial_state: tf.Tensor, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, gamma: float,
               max_steps_per_episode: int) -> tf.Tensor:
    with tf.GradientTape() as tape:
        action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)
        returns = get_expected_return(rewards, gamma)
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        computed_loss = compute_loss(action_probs, values, returns)

    grads = tape.gradient(computed_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return tf.math.reduce_sum(rewards)


def create_model(num_hidden_units, num_actions):
    camera = keras.Input(shape=self_driving_car.STATE_SHAPE)
    vehicle_state = keras.Input(shape=2)

    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(camera)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, vehicle_state])
    x = layers.Dense(units=num_hidden_units, activation='relu')(x)

    actor = layers.Dense(num_actions, activation='softmax')(x)
    critic = layers.Dense(units=1)(x)

    return keras.Model(inputs=[camera, vehicle_state], outputs=[actor, critic])


epsilon = numpy.finfo(numpy.float32).eps.item()
seed = 42
learning_rate = 0.01
gamma = 0.99
min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 800
reward_threshold = 8.0
model_checkpoint = 25

tf.random.set_seed(seed)
numpy.random.seed(seed)
env = gym.make('SelfDrivingCar-v0', render_mode='human', scene_name='env_basic.obj',
               max_episode_steps=max_steps_per_episode)

num_actions = env.action_space.n  # 2
num_hidden_units = 32
model = create_model(num_hidden_units, num_actions)
print(model.summary())
huber_loss = losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile()

# Keep the last episodes reward
episodes_reward = collections.deque(maxlen=min_episodes_criterion)
episodes_trajectory = collections.deque(maxlen=10)

current_time = datetime.now().strftime('%d-%b-%Y_%H-%M-%S')
train_log_dir = 'results/' + current_time + '/train'
test_log_dir = 'results/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

t = tqdm.trange(max_episodes)
for i in t:
    tf.summary.experimental.set_step(i)
    episode = i + 1

    initial_state, info = env.reset()
    # initial_state = tf.ragged.constant(initial_state, dtype=tf.float32)
    episode_reward = float(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))

    episodes_reward.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

    with train_summary_writer.as_default():
        tf.summary.scalar('Reward', episode_reward, step=episode)

    if i % model_checkpoint == model_checkpoint - 1:
        model.save(train_log_dir + '/model.h5')

    if running_reward > reward_threshold and i >= min_episodes_criterion:
        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
        break
