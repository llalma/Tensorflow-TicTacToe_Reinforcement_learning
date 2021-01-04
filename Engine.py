import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import numpy as np
import random


class TicTacToe:
    def __init__(self):
        #Reward Values
        self.rewards = [5, 10, 0]
        self.reward = 0

        #Game Variables
        self.gameOver = False
        self.board = [0]*9
        self.currentPlayer = 1
        self.remainingSquares = [0,1,2,3,4,5,6,7,8]
        self.winner = 0

        # Board tracking
        self.prevState = []
        self.state = []
        self.action = []
        self.over = []
    #end

    def reset(self):
        # Reward Values
        self.reward = 0

        #Game Variables
        self.board = [0]*9
        self.currentPlayer = 1
        self.remainingSquares = [0,1,2,3,4,5,6,7,8]
        self.gameOver = False

        # Board tracking
        self.prevState = []
        self.state = []
        self.action = []
        self.over = []
    #end

    def step(self,action):
        #Play a single round of the game

        #Save previous state of board, before action is performed
        self.prevState.append(self.getBoard())

        #Play a move
        if not self.gameOver:
            self.playMove(action)
        #end

        #Check if game is over
        if self.checkState():
            self.gameOver = True
            self.reward = self.rewards[self.winner]
        #end

        #Switch players after move has been played.
        self.switchPlayers()

        #Reset game for next episode
        # if self.gameOver:
        #     self.reset()
        # #end

        #Save current game state
        self.state.append(self.getBoard())
        self.action.append(action)
        self.over.append(self.gameOver)
    #end

    def getBoard(self):
        return np.array(self.board).reshape(1,9)
    #end

    def getTrainingValues(self):
        #Returns game data for training, in correct format.
        return np.array(self.state).reshape(-1, 9),np.array(self.prevState).reshape(-1, 9),self.action, self.reward, self.over
    #end

    #######################################
    #Game logic
    #######################################

    def playMove(self,action):
        #Attempt to play move, whilst checking square is avaliable
        if action in self.remainingSquares:
            self.board[action] = self.currentPlayer
            self.remainingSquares.remove(action)
        #end
    #end

    def checkState(self):
        #Checks if the game is over in anyway, via a player willing or running out of squares

        #Check for number of squares
        if len(self.remainingSquares) == 0:
            return True
        #end

        # Check if a player won
        if self.checkWinning(1):
            self.winner = 1
            return True
        elif self.checkWinning(2):
            self.winner = 2
            return True
        #end

        #Game is not over so return false
        return False
    #end

    def checkWinning(self, marker):
        if self.board[0] == marker and self.board[1] == marker and self.board[2] == marker or \
                self.board[3] == marker and self.board[4] == marker and self.board[5] == marker or \
                self.board[6] == marker and self.board[7] == marker and self.board[8] == marker or \
                self.board[0] == marker and self.board[3] == marker and self.board[6] == marker or \
                self.board[1] == marker and self.board[4] == marker and self.board[7] == marker or \
                self.board[2] == marker and self.board[5] == marker and self.board[8] == marker or \
                self.board[0] == marker and self.board[4] == marker and self.board[8] == marker or \
                self.board[2] == marker and self.board[4] == marker and self.board[6] == marker:
            return True
        # end
        return False
    # end

    def switchPlayers(self):
        if self.currentPlayer == 1:
            self.currentPlayer = 2
        else:
            self.currentPlayer = 1
        #end
    #end
#end


class Model:
    def __init__(self,path=""):
        if path != "": #Load a pretrained model
            self.model = keras.models.load_model(path)
        else:   #Create model from scratch
            self.model = self.createModel()
        #end

        # Hyper Params
        self.randomChoice = 0.2
        self.alpha = 0.5
        self.gamma = 0.95
        self.targetModel = self.createModel()
    #end

    def createModel(self):
        input = keras.Input(shape=(9), name='input')

        dense1 = layers.Dense(27)(input)
        dense1 = layers.Dense(18)(dense1)

        output = layers.Dense(9, name='output', activation="softmax")(dense1)

        model = keras.Model(inputs=input, outputs=[output])
        model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

        return model
    #end

    def summary(self):
        self.model.summary()
    #end

    def predictMove(self,game,model=""):
        #Gets predictions of moves, then removes impossible moves from predictions

        if model == "":
            model = self.model
        #end

        if type(game) is np.ndarray:
            preds = tf.keras.backend.get_value(model(game.reshape(1,9)))[0]
        else:
            preds = tf.keras.backend.get_value(model(game.getBoard()))[0]

            for i in range(0, len(preds)):
                if i not in game.remainingSquares:
                    preds[i] = 0
                # end
            # end
        #end

        #View prdicted values
        if viewPredictions:
            print(preds)
        #end

        #If playing a real game, dont randomly place
        if play:
            return np.argmax(preds)
        #end

        if (random.randint(0,10)/10) > self.randomChoice:
            return np.argmax(preds)
        else:
            return random.randint(0,8)
        #end
    #end

    def train(self,games):

        for i,game in enumerate(games):
            print("Training game: " + str(i) + "/" + str(len(games)))
            for i in range(len(game.get("over"))):

                target = self.targetModel.predict(np.array(game.get("prevStates")[i]).reshape(1,9))

                if game.get("over")[i]:
                    target[0][game.get("actions")[i]] = game.get("rewards")
                else:
                    qFuture = self.predictMove(game.get("states")[i],self.targetModel)
                    target[0][game.get("actions")[i]] = game.get("rewards") + qFuture * self.gamma
                #end
                self.model.fit(game.get("states")[i].reshape(1,9), target, epochs=1, verbose=0)
            #end
        #end

        self.target_train()
    #end

    def save(self,path):
        print("Saving Model")
        self.model.save(path)
    #end

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.targetModel.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.targetModel.set_weights(target_weights)
    #end
#end

def exampleGame(game,player1,player2):
    #Play an example game bot v bot


    player = True
    game.reset()

    while not game.gameOver:
        if player:
            move = player1.predictMove(game)
            print("Player 1 plays " + str(move))
            game.step(move)
        else:
            move = player2.predictMove(game)
            print("Player 2 plays " + str(move))
            game.step(move)
        print(game.getBoard().reshape(3,3))
        player = not player
    #end
#end

def getBatch(game,player1,player2,batchesToGenerate):
    games = []

    #Get games until got enough data for 1 batch
    while len(games) < batchesToGenerate*batchSize:
        #Play game till its over
        while not game.gameOver and len(game.remainingSquares) > 1:
            game.step(player1.predictMove(game))
            game.step(random.choice(game.remainingSquares))
        #end
        states, prevStates, actions, rewards, over = game.getTrainingValues()

        games.append({
            "states":states,
            "prevStates":prevStates,
            "actions":actions,
            "rewards":rewards,
            "over":over
        })

        game.reset()

        # if len(y)%1000 == 0:
        #     print("Collected " + str(len(y)) + "/" + str(batchesToGenerate*batchSize) + " batches")
        # #end
    #end

    return games

def main():
    global player
    game = TicTacToe()

    model1 = Model("model")
    model1.summary()

    model2 = Model()

    if play:    #Play against the model
        #Loop until game is over
        while not game.gameOver:

            #Show board
            print(game.getBoard().reshape(3, 3))

            #Switch between players
            if player:  #Players turn
                move = int(input("Input digit between 0-8"))
            else:   #Models turn
                move = model1.predictMove(game)
            #end
            player = not player

            #Perform move
            game.step(move)
        #end

    else:   #Train model
        for i in range(epochs):
            games = getBatch(game,model1,model2,batchesToGenerate=batchesToGenerate)

            model1.train(games)
        #end
        model1.save("model")

        exampleGame(game,model1,model2)
    #end
#end

if __name__ == "__main__":
    #Hyper params
    batchSize = 64
    batchesToGenerate = 1
    epochs = 5
    trainVerbose = 1

    play = True    # Set to True to  play against the model
    player = True  # Set to True for real person to play first
    viewPredictions = False

    main()
#end