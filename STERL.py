import enum
import gym
from visualize import *

from stateDescription import *
from getPriceData import *

class Actions(enum.Enum):
    Sell = 0
    Hold = 1
    Buy = 2

class StockImageEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mode = 'Train', symbol='TSLA', runName='storedData', test=True, window=16, startAt=0, howMany= 1):
        self.runName = runName
        # self.visual = visualize(1) # Uncomment this line and render function to start state visualization
        self.symbol = symbol + '_' +mode
        self.window = window
        global RUNTIME
        RUNTIME = datetime.today().strftime('%b_%d_%H_%M_%S_')
        if not test:
            prices = getData(symbol, compact=False, interval=1)
        else:
            print("---Getting "+mode+" Data---")
            prices = readDataPrice(self.symbol)

        dataLength = len(prices)
        howMany = min(howMany + 1, dataLength - 2 * window - startAt)  # limit to what data we have.
        startAt = max(startAt, 2 * window)
        prices = prices[startAt - 2 * window: startAt + howMany]# limit the data to what we will animate over
        self.howMany = howMany

        # precompute the data we need into parallel arrays
        (self.highs, self.lows, self.opens, self.closes) = (getHighs(prices), getLows(prices), getOpens(prices), getCloses(prices))

        self.i = -1
        self.action_space = 3
        # self.tommorowOpen = 0
        # self.currentClose = 0
        self.position = 0 # Initial position is neutral
        self.invalidActionCount = 0

        # print("Initial position: Hold")
        # Initial state will be generated in reset function
        # print("--GETTING INITIAL STATE--")
        # self.currObs = self._generateNewObservation(self.position, Actions.Hold)
        self.currObs = None
        # print("--ENVIRONMENT INITIALISED SUCCESSFULLY--")



    def step(self, action_index, mode = None):
        info = 0
        action = Actions(action_index)

        transaction_fees = 0
        reward = 0
        # Select new position on the basis of previous position and current action taken
        if self.position == 0: # if position is neutral
            if action == Actions.Sell:  # Sell/Short
                self.position = -1.0
                transaction_fees = 0.1

            elif action == Actions.Hold:  # Hold/Nothing
                self.position = 0.0

            elif action == Actions.Buy:  # Buy/Long
                self.position = +1.0
                transaction_fees = 0.1

            reward = self.position * (self.closes[self.t1] - self.opens[self.t1])

        elif self.position == -1: # if position is short
            prevPosition = self.position
            if action == Actions.Sell:  # Sell/Short
                self.position = -1.0
                transaction_fees = 0.5  # INVALID ACTION PENALTY
                self.invalidActionCount += 1
                info = 1
                reward = prevPosition * (self.closes[self.t1] - self.closes[self.t1 - 1])
                print("--------------Invalid action (Selling after selling). Total count: "+ str(self.invalidActionCount)+"!!-------------------")

            elif action == Actions.Hold:  # Hold/Nothing
                self.position = -1.0
                reward = prevPosition * (self.closes[self.t1] - self.closes[self.t1 - 1])

            elif action == Actions.Buy:  # Buy/Long
                self.position = 0.0
                reward = prevPosition * (self.opens[self.t1] - self.closes[self.t1 - 1])

        elif self.position == 1: # if position is long
            prevPosition = self.position
            if action == Actions.Sell:  # Sell/Short
                self.position = 0.0
                reward = prevPosition * (self.opens[self.t1] - self.closes[self.t1 - 1])

            elif action == Actions.Hold:  # Hold/Nothing
                self.position = +1.0
                reward = prevPosition * (self.closes[self.t1] - self.closes[self.t1 - 1])

            elif action == Actions.Buy:  # Buy/Long
                self.position = +1.0
                transaction_fees = 0.5 #INVALID ACTION PENALTY
                self.invalidActionCount += 1
                info = 1
                reward = prevPosition * (self.closes[self.t1] - self.closes[self.t1 - 1])
                print("--------------Invalid action (Buying after buying). Total count: "+ str(self.invalidActionCount)+"!!-------------------")

        # # CALCULATE REWARD FOR THE ACTION TAKEN.
        # self.currentClose = self.closes[self.t1 - 1]
        # self.tommorowOpen = self.opens[self.t1]

        reward = reward - transaction_fees # charging transactions fees

        if info == 1:
            print("--------------------Returning same state. Choose different action.--------------------")
            return self.currObs, reward, 0, info, self.timeIndex
        else:
            #ACTION TAKEN SUCCESSFULLY. NOW GENERATE NEW STATE AND RETURN IT TO THE AGENT AS OBSERVATION.
            obs = self._generateNewObservation(self.position, action)
            return obs, reward, 0, info, self.timeIndex


    def reset(self):
        self.i = -1
        # self.tommorowOpen = 0
        # self.currentClose = 0
        self.position = 0
        self.invalidActionCount = 0
        # print("--RETURNING INITIAL STATE--")
        self.currObs = self._generateNewObservation(self.position, Actions.Hold)
        # print("--ENVIRONMENT INITIALISED SUCCESSFULLY--")
        return self.currObs, 0, 0, 0, self.timeIndex

    def render(self, episode, mode='human', close=False):
        # # Uncomment this along with line 2 in __init__ function to start visualization
        # self.visual.renderState(self.priceMatrix, self.gradients, self.scores, self.window,
        #             scale=0)  # , highs = highs[t0:t1], lows = lows[t0:t1], opens = opens[t0:t1], closes = closes[t0:t1])
        # self.visual.saveImageOne(self.symbol, episode, self.runName, self.i)
        return

    def _generateNewObservation(self, position, action):

        # TICK THE TIME FORWARD
        self.i = self.i + 1


        self.t0 = self.i  # start time is i= np.asarray
        self.t1 = self.t0 + 2 * self.window  # end time is two windows ahead
        self.timeIndex = self.t1 - 1 # Gives the index of current data we are looking at. Used for maintaining action list.

        (self.priceMatrix, self.gradients, self.scores) = generateStateDescription(self.highs[self.t0:self.t1], self.lows[self.t0:self.t1],
                                                                                   self.opens[self.t0:self.t1],
                                                                                   self.closes[self.t0:self.t1], window=self.window)

        # Reshape example: [3, 2, 1] to [[3],[2],[1]] -> Dimension (3,) to (3, 1)

        priceMatrix = self.priceMatrix.reshape(-1, 32, 16, 1)
        gradientImage = np.dstack((self.gradients, self.scores))
        gradientImage = gradientImage.reshape(-1, 32, 16, 2, 1)

        positionArray = [0, 0, 0]
        if position == -1: # if position is short
            positionArray = [1,0,0]

        elif position == 0: # if position is hold
            positionArray = [0,1,0]

        else: # if position is long
            positionArray = [0,0,1]

        actionArray = [0, 0, 0]
        if action == Actions.Sell:  # Sell/Short
            actionArray = [1, 0, 0]

        elif action == Actions.Hold:  # Hold/Nothing
            actionArray = [0, 1, 0]

        else:  # Buy/Long
            actionArray = [0, 0, 1]

        positionArray = np.array(positionArray).reshape(-1,3)
        actionArray = np.array(actionArray).reshape(-1,3)

        # self.currObs = [priceMatrix, gradients, scores, position] # [gradients, scores, position of the agent after taking the action]
        self.currObs = [priceMatrix, gradientImage, positionArray, actionArray]

        return self.currObs




#-----------------------------Below code is used to do experimental testing of the environment--------------------

# howMany will give limit to one episode where as the loop of step will iterate over that episode for no more than howMany states.
# howMany number of states in an episode
# # # # # #
# env = StockImageEnv(mode='Test',howMany=1, symbol='SLGG')
# env.reset()
# # # #
# for time_step in range(1):
#     # print("Timestep--- ",time_step)
#     actions = [0,1,2]
#     doThis = random.choice(actions)
#     # print("=============ACTION NUMBER==============", time_step)
#     # print("Action: ", doThis)
#     state, reward, done, _, _ = env.step(doThis)
#     # print(done)
#     # np.set_printoptions(threshold=np.inf)
#     # print("Shape of observation: ", state) # (600,600,4)
#     # print("-------Reward-----------: ", reward)
#     # print('----------------ONE STEP DONE-----------------')
#     # print(state[-1][0][0])
