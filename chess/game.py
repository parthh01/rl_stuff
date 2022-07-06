import chess 



class Game: 

    def __init__(self):
        self.board = chess.Board()

 

    def state_translation(self):
        """"
        convert given board state into input vector for neural network 
        """
        state = self.board

    