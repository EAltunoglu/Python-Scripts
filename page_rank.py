import math
from copy import deepcopy

TELEPORTATION_RATE = 0.15
TOP_PEOPLE = 50
NAMES = []
WAY = []
VERTEX_COUNT = 0

with open('data.txt', 'r') as fp:
    INPUT_FILE = fp.read().splitlines()

def extract_vertices():
    """
        Extracts vertex list and stores names.
    """
    vertex_count = int(INPUT_FILE[0].split()[1])
    global VERTEX_COUNT, NAMES
    VERTEX_COUNT = vertex_count
    NAMES = [''] * vertex_count
    count = 0
    
    for line in INPUT_FILE[1:]:
        if vertex_count == count:
            break
        line = line.split()
        index = int(line[0]) - 1
        name = line[1][1:len(line[1])-1]
        NAMES[index] = name
        count += 1


def extract_edges():
    """
        Creates adjacency matrix.
        Then turns it to transition matrix.
    """
    global WAY
    for _ in range(VERTEX_COUNT): # initialize adjacency matrix
        WAY.append([0] * VERTEX_COUNT)

    unique_edges = 0
    for line in INPUT_FILE[VERTEX_COUNT+2:]: # process edges
        line = line.split()
        _from = int(line[0]) - 1
        _to = int(line[1]) - 1
        if not WAY[_from][_to]:
            unique_edges += 1
        WAY[_from][_to] = 1.0
        WAY[_to][_from] = 1.0
    
    print("UNIQUE BIDIRECTIONAL EDGES:", unique_edges)

    for i in range(VERTEX_COUNT): # Create transition matrix.
        normalize(WAY[i])
        
        for j in range(VERTEX_COUNT):
            WAY[i][j] = (WAY[i][j] + TELEPORTATION_RATE/VERTEX_COUNT)
        
        normalize(WAY[i])
        assert math.isclose(sum(WAY[i]), 1.0)


def get_initial_matrix() -> list:
    """
        Returns initial probability for each person
    """
    ret: list = []
    ret.append([1.0/VERTEX_COUNT] * VERTEX_COUNT)
    return ret


def matrix_mult(a: list, b: list) -> list:
    """
        Multiplies two matrix.
        Return multiplied matrix.
    """
    assert len(a[0]) == len(b)

    ret = []
    for _ in range(len(a)): # initialize returned matrix
        ret.append([0.0] * len(b[0])) 

    for i in range(len(a)):
        for k in range(len(a[0])):
            for j in range(len(b[0])):
                ret[i][j] += a[i][k] * b[k][j]

    return ret


def normalize(a: list) -> list:
    """
        Normalize probability list by equalizing summation of probs to 1.
    """
    total = sum(a)
    for i in range(len(a)):
        a[i] = a[i] / total


def matrix_expo_until_same() -> list:
    """
        Multiplies initial matrix by transition matrix until no changes happen.
    """
    ans = get_initial_matrix()
    iteration = 0
    
    while True:
        iteration += 1
        prev = deepcopy(ans)
        
        ans = matrix_mult(ans, WAY)
        normalize(ans[0])

        is_same: bool = True # to check similarity
        for i in range(len(ans)):
            is_same = is_same and math.isclose(prev[0][i], ans[0][i])

        if is_same:
            break

    print("ITERATION COUNT:", iteration)
    
    return ans


def matrix_expo(pow: int) -> list:
    """
        Multiply inital matrix by transition matrix 'pow' times.
        Not used in main function.
        It was for testing.
    """
    ans = get_initial_matrix()
    for _ in range(pow):
        print(sum(ans[0]))
        ans = matrix_mult(ans, WAY)
    normalize(ans[0])
    return ans


def matrix_fast_expo(pow: int) -> list:
    """
        Fast exponential multiplication for matrix.
        Not used in main function.
        It was for testing. 
    """
    ans = get_initial_matrix()
    transition = deepcopy(WAY)
    while pow > 0:
        if pow % 2 == 1:
            ans = matrix_mult(ans, transition)
            normalize(ans)
        
        transition = matrix_mult(transition, transition)
        pow = pow // 2

    return ans

def get_top_people(probs: list):
    """
        Prints top 50(TOP_PEOPLE) people with probabilities
    """
    probs = probs[0]
    assert len(probs) == VERTEX_COUNT
    pair_list = []
    
    for i in range(len(probs)):
        pair_list.append((probs[i], i))
    pair_list.sort(reverse=True)
    
    for i in range(TOP_PEOPLE):
        print(NAMES[pair_list[i][1]], pair_list[i][0])
    

def main(args=None):
    extract_vertices()
    extract_edges()

    ans1 = matrix_expo_until_same() # Page Rank
    get_top_people(ans1) # Print results

if __name__ == '__main__':
    main()
