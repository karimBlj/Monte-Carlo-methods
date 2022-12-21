import numpy as np
import copy
import matplotlib.pyplot as plt
from joblib          import Parallel, delayed
from tqdm.notebook   import tqdm
from multiprocessing import Pool
from functools       import partial

def add (board):
    nplayouts = [0.0 for x in range (4)]
    nwins = [0.0 for x in range (4)] 
    Table [board.encode_board()] = [0, nplayouts, nwins]
 
def look (board):
    return Table.get (board.encode_board(), None)

def UCT (board, C=0.4):
    if board.game_over ():
        return board.score
    t = look (board)
    if t != None:
        bestValue = np.NINF
        best = 0
        moves = board.available_moves
        for i in range (0, len (moves)):
            val = np.Inf
            if t [1] [i] > 0:
                Q = t [2] [i] / t [1] [i]
                val = Q + C * np.sqrt (np.log (t [0]) / t [1] [i])
            if val > bestValue:
                bestValue = val
                best = i
        board.play (moves [best])
        res = UCT (board, C)
        t [0] += 1
        t [1] [best] += 1
        t [2] [best] += res
        return res
    else:
        add (board)
        return board.playout ()


def BestMoveUCT (board, n, C=0.4):
    global Table
    Table = {}
    for i in tqdm(range(n)):
        b1 = copy.deepcopy(board)
        res = UCT(b1, C)
    t = look(board)
    moves = board.available_moves
    best = moves[0]
    bestValue = t[1][0]
    for i in range(1, len(moves)):
        if (t[1][i] > bestValue):
            bestValue = t[1][i]
            best = moves[i]
    return best

def playoutAMAF (board, played):
    while (True):
        moves = []
        moves = board.available_moves
        if len(moves) == 0 or board.game_over():
            return board.score
        n = np.random.randint(0, len(moves))
        played.append(Board.code[moves[n]])
        board.play(moves[n])

def addAMAF (board):
    nplayouts = [0.0 for x in range(4)]
    nwins = [0.0 for x in range(4)]
    nplayoutsAMAF = [0.0 for x in range(4)]
    nwinsAMAF = [0.0 for x in range(4)]
    Table [board.encode_board()] = [0, nplayouts, nwins, nplayoutsAMAF, nwinsAMAF]

def updateAMAF (t, played, res):
    for i in range(len(played)):
        code = played [i]
        seen = False
        for j in range (i):
            if played [j] == code:
                seen = True
        if not seen:
            t[3][code] += 1
            t[4][code] += res

def RAVE (board, played, bias=1e-5):
    if (board.game_over()):
        return board.score
    t = look(board)
    if t != None:
        bestValue = np.NINF
        best = 0
        moves = board.available_moves
        bestcode = Board.code[moves[0]]
        for i in range (0, len (moves)):
            val = np.Inf
            code = Board.code[moves[i]]
            if t [3] [code] > 0:
                beta = t [3] [code] / (t [1] [i] + t [3] [code] + bias * t [1] [i] * t [3] [code])
                Q = 1
                if t [1] [i] > 0:
                    Q = t [2] [i] / t [1] [i]
                AMAF = t [4] [code] / t [3] [code]
                val = (1.0 - beta) * Q + beta * AMAF
            if val > bestValue:
                bestValue = val
                best = i
                bestcode = code
        board.play (moves [best])
        played.append (bestcode)
        res = RAVE (board, played, bias)
        t [0] += 1
        t [1] [best] += 1
        t [2] [best] += res
        updateAMAF (t, played, res)
        return res
    else:
        addAMAF (board)
        return playoutAMAF (board, played)


def BestMoveRAVE (board, n, bias=1e-5):
    global Table
    Table = {}
    for i in tqdm(range (n)):
        b1 = copy.deepcopy (board)
        res = RAVE (b1, [], bias)
    t = look (board)
    moves = board.available_moves
    best = moves [0]
    bestValue = t [1] [0]
    for i in range (1, len(moves)):
        if (t [1] [i] > bestValue):
            bestValue = t [1] [i]
            best = moves [i]
    return best

def GRAVE (board, played, tref, bias=1e-5):
    if (board.game_over()):
        return board.score
    t = look (board)
    if t != None:
        tr = tref
        if t [0] > 50:
            tr = t
        bestValue = np.NINF
        best = 0
        moves = board.available_moves
        bestcode = Board.code[moves[0]]
        for i in range (0, len (moves)):
            val = np.Inf
            code = Board.code[moves[i]]
            if tr [3] [code] > 0:
                beta = tr [3] [code] / (t [1] [i] + tr [3] [code] + bias * t [1] [i] * tr [3] [code])
                Q = 1
                if t [1] [i] > 0:
                    Q = t [2] [i] / t [1] [i]
                AMAF = tr [4] [code] / tr [3] [code]
                val = (1.0 - beta) * Q + beta * AMAF
            if val > bestValue:
                bestValue = val
                best = i
                bestcode = code
        board.play (moves [best])
        played.append (bestcode)
        res = GRAVE (board, played, tr, bias)
        t [0] += 1
        t [1] [best] += 1
        t [2] [best] += res
        updateAMAF (t, played, res)
        return res
    else:
        addAMAF (board)
        return playoutAMAF (board, played)

    
def BestMoveGRAVE (board, n, bias=1e-5):
    global Table
    Table = {}
    addAMAF (board)
    for i in range (n):
        t = look (board)
        b1 = copy.deepcopy (board)
        res = GRAVE (b1, [], t, bias)
    t = look (board)
    moves = board.available_moves
    best = moves [0]
    bestValue = t [1] [0]
    for i in range (1, len(moves)):
        if (t [1] [i] > bestValue):
            bestValue = t [1] [i]
            best = moves [i]
    return best

def UCTNested (board, t1, C=0.4):
    if board.game_over():
        return board.misereScore () / (t1 + 1)
        # return board.score / (t1 + 1)
    t = look(board)
    if t != None:
        bestValue = np.NINF
        best = 0
        moves = board.available_moves
        for i in range (0, len(moves)):
            val = np.Inf
            if t [1] [i] > 0:
                Q = t [2] [i] / t [1] [i]
                # if board.turn == Black:
                #     Q = -Q
                val = Q + C * np.sqrt (np.log (t [0]) / t [1] [i])
            if val > bestValue:
                bestValue = val
                best = i
        board.play (moves [best])
        res = UCTNested (board, t1 + 1)
        t [0] += 1
        t [1] [best] += 1
        t [2] [best] += res
        return res
    else:
        add (board)
        return board.nestedDiscountedPlayout (t1)