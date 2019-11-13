import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from scores.score_logger import ScoreLogger

# options
do_render = False # set to True to show the gym environment during training

# misc constants
ENV_NAME = "CartPole-v1" # name of the gym environment to use

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

# exploration values
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

# success metrics
AVERAGE_SCORE_TO_SOLVE = 195
CONSECUTIVE_RUNS_TO_SOLVE = 100

class DQNSolver:

	def __init__(self, observation_space, action_space, exploration_max, exploration_min, exploration_decay, memory_size, batch_size, learning_rate, gamma):
		# initialize exploration rate to the max value, we will decay it over time
		self.exploration_rate = exploration_max
		self.exploration_min = exploration_min
		self.exploration_decay = exploration_decay

		self.action_space = action_space
		self.memory = deque(maxlen=memory_size)
		self.batch_size = batch_size
		self.gamma = gamma

		self.model = Sequential()
		self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
		self.model.add(Dense(24, activation="relu"))
		self.model.add(Dense(self.action_space, activation="linear"))
		self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate))

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() < self.exploration_rate:
			return random.randrange(self.action_space)
		q_values = self.model.predict(state)
		return np.argmax(q_values[0])

	def experience_replay(self):
		if len(self.memory) < self.batch_size:
			return
		batch = random.sample(self.memory, self.batch_size)
		for state, action, reward, state_next, terminal in batch:
			q_update = reward
			if not terminal:
				q_update = (reward + self.gamma * np.amax(self.model.predict(state_next)[0]))
			q_values = self.model.predict(state)
			q_values[0][action] = q_update
			self.model.fit(state, q_values, verbose=0)
		self.exploration_rate *= self.exploration_decay
		self.exploration_rate = max(self.exploration_min, self.exploration_rate)


def cartpole():
	env = gym.make(ENV_NAME)
	score_logger = ScoreLogger(ENV_NAME, AVERAGE_SCORE_TO_SOLVE, CONSECUTIVE_RUNS_TO_SOLVE)
	observation_space = env.observation_space.shape[0]
	action_space = env.action_space.n
	dqn_solver = DQNSolver(observation_space, action_space, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY, MEMORY_SIZE, BATCH_SIZE, LEARNING_RATE, GAMMA)

	run = 0
	while True:
		run += 1
		state = env.reset()
		state = np.reshape(state, [1, observation_space])
		step = 0
		while True:
			step += 1
			if do_render:
				env.render()
			action = dqn_solver.act(state)
			state_next, reward, terminal, info = env.step(action)
			reward = reward if not terminal else -reward
			state_next = np.reshape(state_next, [1, observation_space])
			dqn_solver.remember(state, action, reward, state_next, terminal)
			state = state_next
			if terminal:
				print(f"Run: {run}, exploration: {dqn_solver.exploration_rate}, score: {step}")
				score_logger.add_score(step, run)
				break
			dqn_solver.experience_replay()


if __name__ == "__main__":
	cartpole()
