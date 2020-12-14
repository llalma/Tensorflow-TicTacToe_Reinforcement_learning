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
        self.state = [self.getBoard()]
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
        self.state = [self.getBoard()]
    #end

    def step(self,action):
        #Play a single round of the game

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
    #end

    def getBoard(self):
        return np.array(self.board).reshape(1,9)
    #end

    def getTrainingValues(self):
        #Returns game data for training, in correct format.
        return np.array(self.state).reshape(-1, 9), self.reward
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
            input = keras.Input(shape=(9), name='input')

            dense1 = layers.Dense(27)(input)
            dense1 = layers.Dense(18)(dense1)

            output = layers.Dense(9, activation='softmax', name='output')(dense1)

            self.model = keras.Model(inputs=input, outputs=[output])
            self.model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['accuracy'])
        #end

        # Hyper Params
        self.randomChoice = 0.2
        self.alpha = 0.5
    #end

    def summary(self):
        self.model.summary()
    #end

    def predictMove(self,game):
        #Gets predictions of moves, then removes impossible moves from predictions

        if type(game) is np.ndarray:
            preds = tf.keras.backend.get_value(self.model(game.reshape(1,9)))[0]
        else:
            preds = tf.keras.backend.get_value(self.model(game.getBoard()))[0]

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

        if (random.randint(0,10)/10) > self.randomChoice:
            return np.argmax(preds)
        else:
            return random.randint(0,8)
        #end
    #end

    def train(self,x,y):
        x_train = []
        y_train = []

        # Get target values for training
        for game,R in zip(x,y):
            prevState = game[0]

            for state in game[1:]:
                prevPrediction = self.predictMove(prevState)
                currPrediction = self.predictMove(state)

                x_train.append(prevState)
                y_train.append(np.array(prevPrediction + self.alpha * (R[0] + currPrediction - prevPrediction)))

                prevState = state
            #end
        #end

        self.model.fit(x=np.array(x_train),y=np.array(y_train),batch_size=batchSize,epochs=1,verbose=trainVerbose)
    #end

    def save(self,path):
        self.model.save(path)
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
    x = []
    y = []

    #Get games until got enough data for 1 batch
    while len(y) < batchesToGenerate*batchSize:
        #Play game till its over
        while not game.gameOver and len(game.remainingSquares) > 1:
            game.step(player1.predictMove(game))
            game.step(random.choice(game.remainingSquares))
        #end
        currentX, currentY = game.getTrainingValues()
        x.append(currentX)
        y.append(currentY)

        game.reset()

        if len(y)%1000 == 0:
            print("Collected " + str(len(y)) + "/" + str(batchesToGenerate*batchSize) + " batches")
        #end
    #end

    return x, np.array(y).reshape(batchesToGenerate*batchSize,1)
#end

def main():
    global player
    game = TicTacToe()

    model1 = Model()
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
            x,y = getBatch(game,model1,model2,batchesToGenerate=batchesToGenerate)

            model1.train(x,y)
            print(y[0])
        #end
        model1.save("model")

        exampleGame(game,model1,model2)
    #end
#end

if __name__ == "__main__":
    #Hyper params
    batchSize = 64
    batchesToGenerate = 5
    epochs = 10
    trainVerbose = 1

    play = False    # Set to True to  play against the model
    player = True  # Set to True for real person to play first
    viewPredictions = False

    main()
#end