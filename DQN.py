import random
from collections import deque
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Input, Dropout, MaxPooling2D
from keras.optimizers import RMSprop

from utils import timeit
from keras.backend import cast
from plotReward import *
from STERL import *
from saveActions import saveAction
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Parameters

gamma = 0.95  # Discount factor
learning_rate = 0.00025  # Learning rate

memory_size = 1000000  # Memory pool for experience replay, forgets older values as the size is exceeded
batch_size = 16  # Batch size for random sampling in the memory pool

exploration_max = 1.0  # Initial exploration rate
exploration_min = 0.1  # Min value of exploration rate post decay
exploration_decay = 0.9995  # Exploration rate decay rate


class DQNSolver:

    def __init__(self, action_space, mode):
        self.mode = mode
        print('Instantiating the network...')

        # Keras callback used to save weights of the NN
        self.checkpoint_path = "CNN_Model/CNN_Model.ckpt"
        self.cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                           save_weights_only=True,
                                                           verbose=0)

        self.exploration_rate = exploration_max  # Sets initial exploration rate to max
        # self.old_episode = 0  # To check for ep changes (needed to decay exploration rate)

        self.action_space = action_space  # Action space = 3 (Sell, Hold, Buy)
        self.memory = deque(maxlen=memory_size)  # Will forget old values as new ones are appended

        # Defining the network structure
        self.q_model = self.initializeModel()
        self.target_q_model = self.initializeModel()
        # print(self.model.summary())

        if self.mode == 'Test':
            # Loading weights here
            print('Loading weights...')
            self.exploration_rate = 0
            self.q_model.load_weights(self.checkpoint_path)

    def updateTargetWeights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())
        print("----Weights copied from Q-Model to Target-Q-Model----")

    # Multi input neural network initialization
    def initializeModel(self):
        # Create the input vectors
        image_input1 = Input(shape=(32, 16, 1), name="Image1_input") # Price
        image_input2 = Input(shape=(32, 16, 2, 1), name="Image2_input")
        # image_input3 = keras.layers.Input(shape=image_shape3, name="ImageInput3")
        float_input1 = Input(shape=(3,), name="Float_input")
        float_input2 = Input(shape=(3,), name="Float_input2")

        # Build the first convolutional model
        image_network1 = Conv2D(32, 8, strides=(2, 2), padding="same", activation="relu")(image_input1)
        image_network1 = Conv2D(64, 4, strides=(2, 2), padding="same", activation="relu")(image_network1)
        image_network1 = Conv2D(32, 3, strides=(2, 2), padding="same", activation="relu")(image_network1)
        image_network1 = Flatten()(image_network1)
        image_network1 = Dense(32, activation="relu")(image_network1)

        # Build the second convolutional model
        image_network2 = Conv3D(32, 8, strides=2, padding="same", activation="relu")(image_input2)
        image_network2 = Conv3D(64, 4, strides=2, padding="same", activation="relu")(image_network2)
        image_network2 = Conv3D(32, 3, strides=2, padding="same", activation="relu")(image_network2)
        image_network2 = Flatten()(image_network2)
        image_network2 = Dense(32, activation="relu")(image_network2)

        # # Build the third convolutional model
        # image_network3 = Conv2D(32, 8, strides=(2, 2), padding="same", activation="relu")(image_input3)
        # image_network3 = Conv2D(64, 4, strides=(2, 2), padding="same", activation="relu")(image_network3)
        # image_network3 = Conv2D(32, 3, strides=(2, 2), padding="same", activation="relu")(image_network3)
        # image_network3 = Flatten()(image_network3)
        # image_network3 = Dense(32, activation="relu")(image_network3)

        # Build the first dense model
        float_network1 = Dense(4, activation="relu", name="float_layer")(float_input1)
        float_network1 = Dense(4, activation="relu")(float_network1)

        float_network2 = Dense(4, activation="relu", name="float_layer2")(float_input2)
        float_network2 = Dense(4, activation="relu")(float_network2)

        # Use the outputs of the four input models to create a new model called "merged_network"
        concatenate_input = keras.layers.Concatenate(axis=-1, name='merged_layer')([
            cast(image_network1, 'float32'),
            cast(image_network2, 'float32'),
            cast(float_network1, 'float32'),
            cast(float_network2, 'float32')
        ])
        merged_network = Dense(128, activation="relu")(concatenate_input)
        merged_network = Dense(64, activation="relu")(merged_network)
        merged_network = Dense(units=3)(merged_network)

        # Feed the three input models into the one output model
        model = Model(
            inputs=[image_input1, image_input2, float_input1, float_input2],
            outputs=[merged_network]
        )
        model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01),
                      metrics=["accuracy"])
        # print(model.summary())
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))  # Remembering instances in memory for future use

    def act(self, state):
        # Random action -> 0, 1, 2
        if np.random.rand() < self.exploration_rate:
            return np.random.rand(self.action_space)

        # Q values based on model prediction on current state (Initially based on random weights)
        q_values = self.q_model.predict(state)  # Returns an array of prediction eg: [[value1, value2, value3]]
        return q_values[0]  # Tuple of 3 Q values, one for each action

    def experience_replay(self, episode):

        if len(self.memory) < batch_size:  # Not enough memory, do nothing
            return

        # If has enough memory obtained, perform random batch sampling among those
        batch = random.sample(self.memory, batch_size)  # Get a random batch
        for state, action, reward, state_next in batch:
            # Obtain Q value based on immediate reward and predicted q* value of next state using DDQN algorithm
            # DDQN --> Y_t ≡ R_t+1 +γQ(S_t+1 , argmax Q(S_t+1 , a; θ_t ), θ'_t )
            next_action = np.argmax(self.q_model.predict(state_next)[0])
            q_max = self.target_q_model.predict(state_next)[0][next_action]
            q_update = reward + gamma * q_max
            q_values = self.q_model.predict(state)  # Obtain q value tuple for that state
            q_values[0][action] = q_update  # Update the q value for that state, action (one that we took)
            # Update the weights of the network based on the updated q value (based on immediate reward)
            self.q_model.fit(state, q_values, epochs=1, verbose=0, callbacks=[self.cp_callback])

        # Decaying exploration rate at every step
        self.exploration_rate = self.exploration_rate * exploration_decay  # Decay exploration rate
        self.exploration_rate = max(exploration_min, self.exploration_rate)  # Do not go below the minimum

        # # To decay the exploration rate if the episode changes
        # if episode != self.old_episode:
        #     self.exploration_rate = self.exploration_rate * exploration_decay  # Decay exploration rate
        #     self.exploration_rate = max(exploration_min, self.exploration_rate)  # Do not go below the minimum
        # self.old_episode = episode


@timeit
def DQN_Agent(mode, episodes, trainSteps, testSteps, symbol):
    if mode == 'Train':
        train_plotter = plotReward("Train", title='Training')  # For plotting train reward
        # Give testing = False if you don't want to test after every train episode
        train(episodes=episodes, trainSteps=trainSteps, train_plotter=train_plotter, symbol=symbol, testSteps=testSteps,
              testing=True)
    if mode == 'Test':
        test_plotter = plotReward("Test", title='Testing')  # For plotting test reward
        test(testSteps=testSteps, test_plotter=test_plotter, symbol=symbol)


def train(episodes, trainSteps, train_plotter, symbol, testSteps, testing=False):
    mode = 'Train'
    episodes = episodes
    steps = trainSteps
    if testing:
        test_plotter = plotReward("Test", title='Testing')

    print('\nTraining...\n')

    # DQN Stocks
    env = StockImageEnv(mode=mode, howMany=steps, symbol=symbol)  # Object of the environment

    # Get action space
    action_space = env.action_space

    # Object for the solver
    dqn_solver = DQNSolver(action_space, mode)

    episode = 1
    score = [0]

    # Running for a number of episodes
    while episode <= episodes:
        dqn_solver.updateTargetWeights()  # Update the weights of target-q-model at the end of each episode
        #  Resetting initial state, step size, cumulative reward and storing arrays at the start of each episode
        state, _, _, _, _ = env.reset()  # Get initial state
        step = 1
        cumulative_reward = 0

        while step <= steps:
            # env.render(episode)
            actionArray = dqn_solver.act(state)
            action = np.argmax(actionArray)  # Get action based on argmax of the Q value approximation from the NN
            state_next, reward, _, info, _ = env.step(action, mode)

            if info == 1:  # Check if the action was invalid. Do another action if thats the case.
                dqn_solver.remember(state, action, reward,
                                    state_next)  # Remember the invalid action to learn not to repeat it.
                action = np.argsort(actionArray)[-2]  # Choosing second best action.
                state_next, reward, _, info, NextTimeIndex = env.step(action,
                                                                      mode)  # New state after taking different action.
                print("Action changed.")

            cumulative_reward += reward

            if action == 0:
                action_actual = 'Sell'
            if action == 1:
                action_actual = 'Hold'
            if action == 2:
                action_actual = 'Buy'

            print("{} {}ing:".format(step, action_actual) + "\tReward = " + "{:.2f}".format(
                reward) + "\tCumulative reward = "
                          "{:.2f}".format(cumulative_reward))

            dqn_solver.remember(state, action, reward, state_next)  # Remember this instance
            state = state_next  # Update the state
            dqn_solver.experience_replay(episode)  # Perform experience replay to update the network weights
            train_plotter.addToPlot(step, cumulative_reward)  # Add step and cumulative reward to plot for visualization
            if step == steps:
                score.append(cumulative_reward)
                train_plotter.incrementEpisodePlot()
                break
            else:
                step += 1

        print("--------Episode: " + str(episode) + ". Net Reward : " + "{:.2f}".format(score[episode]) + "-------")
        if testing == True:
            test(testSteps, test_plotter, symbol)
        episode += 1


def test(testSteps, test_plotter, symbol):
    mode = 'Test'
    test_episodes = 1
    test_steps = testSteps

    # DQN Stocks
    env = StockImageEnv(mode=mode, howMany=test_steps, symbol=symbol)  # Resetting the environment

    # Get action space
    action_space = env.action_space

    # Object for the solver
    dqn_solver = DQNSolver(action_space, mode)

    episode = 1
    score = [0]

    # Running for a number of episodes
    while episode <= test_episodes:
        #  Resetting initial state, step size, cumulative reward and storing arrays at the start of each episode
        state, _, _, _, CurrTimeIndex = env.reset()  # Get initial state
        step = 1
        cumulative_reward = 0

        # To append step, cumulative reward, corresponding action to plot for each episode
        time_index = []  # For saving action taken and its index
        actions_taken = []

        while step <= test_steps:
            # env.render(episode)
            actionArray = dqn_solver.act(state)
            action = np.argmax(actionArray)  # Get action based on argmax of the Q value approximation from the NN
            state_next, reward, _, info, NextTimeIndex = env.step(action, mode)

            if info == 1:  # Check if the action was invalid. Do another action if thats the case.
                action = np.argsort(actionArray)[-2]  # Choosing second best action.
                state_next, reward, _, info, NextTimeIndex = env.step(action,
                                                                      mode)  # New state after taking different action.
                print("Action changed.")

            cumulative_reward += reward

            if action == 0:
                action_actual = 'Sell'
            if action == 1:
                action_actual = 'Hold'
            if action == 2:
                action_actual = 'Buy'

            time_index.append(CurrTimeIndex)  # For saving action taken and its index
            actions_taken.append(action)

            print("{} {}ing:".format(step, action_actual) + "\tReward = " + "{:.2f}".format(
                reward) + "\tCumulative reward = "
                          "{:.2f}".format(cumulative_reward))

            state = state_next  # Update the state
            CurrTimeIndex = NextTimeIndex  # Update time Index

            test_plotter.addToPlot(step, cumulative_reward)

            if step == test_steps:
                saveAction(time_index, actions_taken, mode, episode)
                score.append(cumulative_reward)
                test_plotter.incrementEpisodePlot()
                break
            else:
                step += 1

        print("Testing Episode: Score : " + "{:.2f}".format(score[episode]))
        print("----Going out of Testing----")
        episode += 1


if __name__ == "__main__":
    train_steps = 700
    test_steps = 700
    episodes = 10

    mode = 'Train'
    # mode = 'Test'
    DQN_Agent(mode, episodes, train_steps, test_steps, symbol='TSLA')
