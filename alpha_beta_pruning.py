import sys
import copy

MIN_VALUE = -10000000
MAX_VALUE = 10000000
MAX_PLAYER = 1
MIN_PLAYER = 2
N = 0
M = 0
BEST_MOVE = ''

DIRECTIONS = [[-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1]]
DIRECTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
DIRECTION_LEN = 8
utility_function_call_count = 0

def is_moveable(x: int, y: int, dir: int) -> bool:
    return x + DIRECTIONS[dir][0] >= 0 and x + DIRECTIONS[dir][0] < N \
        and y + DIRECTIONS[dir][1] >= 0 and y + DIRECTIONS[dir][1] < M 

def is_correct_dir(piece: str, dir: int):
    if piece == 'Q':
        return True
    if piece == 'R' and dir%2 == 0:
        return True
    if piece == 'B' and dir%2 == 1:
        return True
    return False

def get_moved_pos(board: list, x: int, y: int, dir: int) -> tuple:
    while is_moveable(x, y, dir):
        x += DIRECTIONS[dir][0]
        y += DIRECTIONS[dir][1]

        if board[x][y] != 'x':
            return x, y

    return x, y

def max_value(state: list, depth: int, is_pruning: bool, is_random: bool, is_initial_state: bool, alpha=MIN_VALUE, beta=MAX_VALUE):
    if depth == 0:
        return utility_function(state)
    
    result = MIN_VALUE
    
    for i in range(N):
        for j in range(M):
            pos = state[i][j]
            if pos == 'x' or int(pos[1]) != MAX_PLAYER:
                continue
            
            for k in range(DIRECTION_LEN):
                if is_moveable(i, j, k) and is_correct_dir(pos[0], k):
                    x, y = get_moved_pos(state, i, j, k)
                    next_board = copy.deepcopy(state)
                    next_board[x][y] = next_board[i][j]
                    next_board[i][j] = 'x'

                    val = min_value(next_board, depth-1, is_pruning, is_random, alpha, beta)
                    
                    if val > result:
                        result = val

                        if is_initial_state:
                            global BEST_MOVE
                            BEST_MOVE = next_board[x][y][0] + ' ' + DIRECTION_NAMES[k]

                    if is_pruning:
                        if result >= beta:
                            return result
                        
                        alpha = max(alpha, result)

    return result

def min_value(state: list, depth: int, is_pruning: bool, is_random: bool, alpha=MIN_VALUE, beta=MAX_VALUE):
    if is_random:
        result = 0
    else:
        result = MAX_VALUE
    
    move_count = 0

    for i in range(N):
        for j in range(M):
            pos = state[i][j]
            if pos == 'x' or int(pos[1]) != MIN_PLAYER:
                continue
            
            for k in range(DIRECTION_LEN):
                if is_moveable(i, j, k) and is_correct_dir(pos[0], k):
                    x, y = get_moved_pos(state, i, j, k)
                    next_board = copy.deepcopy(state)
                    next_board[x][y] = next_board[i][j]
                    next_board[i][j] = 'x'

                    move_count += 1
                    if is_random:
                        result += max_value(next_board, depth-1, is_pruning, is_random, False, alpha, beta)
                    else:
                        result = min(result, max_value(next_board, depth-1, is_pruning, is_random, False, alpha, beta))
                    
                    if is_pruning:
                        if result <= alpha:
                            return result

                        beta = min(beta, result)

    if is_random:
        result /= move_count
    
    return result

def utility_function(board: list):
    global utility_function_call_count
    utility_function_call_count += 1

    piece_counts = {
        'Q': 0,
        'R': 0,
        'B': 0,
    }

    for row in board:
        for pos in row:
            if pos == 'x':
                continue
            piece_counts[pos[0]] += 1 if pos[1] == '1' else -1

    return 9 * piece_counts['Q'] + 5 * piece_counts['R'] + 3 * piece_counts['B']


def main(args):
    algorithm = args[0]
    input_file = open(args[1], 'r')
    depth = int(args[2])

    lines = input_file.readlines()
    start_pos = []
    
    lines = lines[1:]
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        start_pos.append(line)

    global N, M
    N = len(start_pos)
    M = len(start_pos[0])

    is_random = algorithm == 'minimax_rand'
    is_pruning = algorithm == 'alpha_beta_pruning'

    answer = max_value(start_pos, depth+depth, is_pruning, is_random, True)

    print('Action: Move', BEST_MOVE)
    print('Value:', "{:.2f}".format(answer))
    print('Util calls:', utility_function_call_count)

if __name__ == '__main__':
    main(sys.argv[1:])