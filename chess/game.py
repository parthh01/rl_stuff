import chess 
import numpy as np 


class Game: 

    """
    Important: 
        - pychess board squares are 0 indexed.
        - to convert given board state into input vector for neural network. 
            we need to concat the following: 
            ================================
            piece bits (33 pcs = 6 bits )
            =============================== per square 
            condition bits (6 possible conditions = 3 bits)
            player bit (2 possible players = 1 bit)

    """


    def __init__(self):
        self.board = chess.Board()

        self.N = 64 #board size 
        self.piece_representation = {
            #piece representation for black, add max() from this dict to the piece for black 
            'p': 1, 
            'r': 2, 
            'n': 3, 
            'b': 4, 
            'q': 5, 
            'k': 6,
            'rc': 7, #empty square that is available for en passant 
            'e': 8,
            'ep': 9, #empty square 
        }

    def state_serialization(self):
        """
        input: board state -> output: 64 x 14 matrix of bit-wise representation 
        """
        en_passant = self.board.ep_square
        white_offset = max(self.piece_representation.values()) 
        A = np.zeros(self.N,np.uint8)
        for i in range(self.N):
            val = 0
            piece = self.board.piece_at(i)
            if piece is None: 
                val = self.piece_representation['ep'] if en_passant == i else self.piece_representation['e']
            else: 
                symbol = piece.symbol().lower()
                val = self.piece_representation[symbol]
                if piece.color: val += white_offset
            
            A[i] = val 
    
        if self.board.has_queenside_castling_rights(chess.WHITE) and (A[0] == (self.piece_representation['r'] + white_offset)):
            A[0] = self.piece_representation['rc'] + white_offset
        if self.board.has_kingside_castling_rights(chess.WHITE) and (A[7] == (self.piece_representation['r'] + white_offset)):
            A[7] = self.piece_representation['rc'] + white_offset
        if self.board.has_queenside_castling_rights(chess.BLACK) and (A[56] == self.piece_representation['r']):
            A[56] = self.piece_representation['rc'] 
        if self.board.has_kingside_castling_rights(chess.BLACK) and (A[63] == self.piece_representation['r'] ):
            A[63] = self.piece_representation['rc'] 

        A = A.reshape((8,8))

        ## straight lifted from geohot, using bitshift was too smart 
            # binary state
        binarized = np.zeros((5,8,8),np.uint8)

        # 0-3 columns to binary
        binarized[0] = (A>>3)&1
        binarized[1] = (A>>2)&1
        binarized[2] = (A>>1)&1
        binarized[3] = (A>>0)&1

        # 4th column is who's turn it is
        binarized[4] = (self.board.turn*1.0) 
        #I think just drop the matrix (and this last col) and make it a 257 bit input FC network
        # is conv net really needed? 

        return binarized.reshape((1,5,8,8))


    def action_space(self):
        return list(self.board.legal_moves)
    
    @staticmethod
    def reward_function(board):
        if board.outcome() is None: 
            return 0 
        res = board.outcome().result()
        return eval(res) #1-0 means win for white 
                

    def play(self,ai,is_ai_white=False):

        self.board.reset()
        is_ai_turn = is_ai_white

        while self.board.outcome() is None:
            print(self.board)
            print("=======================")
            move = ai.policy_evaluation(self) if is_ai_turn else input(f"Moving as {'White' if self.board.turn else 'Black'}. Enter move in SAN here: ") 
            self.board.push(move) if is_ai_turn else self.board.push_san(move)
            is_ai_turn = not is_ai_turn
        
        print(self.board)
        print(f"GAME OVER {self.board.outcome.result()}")




                    

            






if __name__ == "__main__":
    game = Game() 
    serial = game.state_serialization()
    print(game.board)
    print(serial)
    print(serial.shape)
    b = chess.Board('8/K7/8/1k6/8/8/8/8 w - - 0 72')
    print(game.reward_function(b))


    