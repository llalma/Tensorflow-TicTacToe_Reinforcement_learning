import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import numpy as np


class TicTacToe:
    def __init__(self):
        #Reward Values
        self.rewards = [0, 100, -100]
        self.reward = 0

        #Game Variables
        self.gameOver = False
        self.board = [0]*9
        self.currentPlayer = 1
        self.remainingSquares = [0,1,2,3,4,5,6,7,8]
        self.winner = 0
    #end

    def reset(self):
        # Reward Values
        self.reward = 0

        #Game Variables
        self.board = [0]*9
        self.currentPlayer = 1
        self.remainingSquares = [0,1,2,3,4,5,6,7,8]
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
        if self.gameOver:
            self.reset()
        #end
    #end

    def getBoard(self):
        return np.array(self.board).reshape(1,9)
    #end

    def getTrainingValues(self):
        #Returns game data for training, in correct format.
        return np.array(self.board).reshape(1,9), self.reward
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

            dense1 = layers.Dense(50)(input)

            output = layers.Dense(9, activation='sigmoid', name='output')(dense1)

            self.model = keras.Model(inputs=input, outputs=[output])
            self.model.compile(loss="huber", optimizer="adam")
        #end
    #end

    def summary(self):
        self.model.summary()
    #end

    def predictMove(self,game):
        #Gets predictions of moves, then removes impossible moves from predictions

        preds = tf.keras.backend.get_value(self.model(game.getBoard()))[0]

        for i in range(0, len(preds)):
            if i not in game.remainingSquares:
                preds[i] = 0
            #end
        #end

        return np.argmax(preds)
    #end

    def train(self,x,y):
        self.model.fit(x=x,y=y,batch_size=batchSize,epochs=epochs,verbose=trainVerbose)
    #end

    def save(self,path):
        self.model.save(path)
    #end
#end

def getBatch(game,player,batchesToGenerate):
    x = []
    y = []

    #Get games until got enough data for 1 batch
    while len(y) < batchesToGenerate*batchSize:
        #Play game till its over
        while not game.gameOver:
            move = player.predictMove(game)
            game.step(move)
        #end
        tempX, tempY = game.getTrainingValues()

        x.append(tempX)
        y.append(tempY)
    #end

    return np.array(x).reshape(batchesToGenerate*batchSize,9), np.array(y).reshape(batchesToGenerate*batchSize,1)
#end

def main():
    global player
    game = TicTacToe()

    model = Model("model")
    model.summary()

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
                move = model.predictMove(game)
            #end
            player = not player

            #Perform move
            game.step(move)
        #end

    else:   #Train model

        x,y = getBatch(game,model,batchesToGenerate=batchesToGenerate)
        model.train(x,y)

        model.save("model")
    #end
#end

if __name__ == "__main__":
    #Hyper params
    batchSize = 128
    batchesToGenerate = 100
    epochs = 20
    trainVerbose = 1

    play = True    # Set to True to  play against the model
    player = True  # Set to True for real person to play first

    main()
#end