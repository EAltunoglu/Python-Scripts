def smith_waterman_algorithm(seq1, seq2, penalty_params):
    v = len(seq1)
    w = len(seq2)
    # Create matrices of zeros for the dynamic programming
    dp = [[0 for _ in range(w+1)] for __ in range(v+1)] # main dynamic programming matrix
    dp_1 = [[0 for _ in range(w+1)] for __ in range(v+1)] # for already opened a gap in first string
    dp_2 = [[0 for _ in range(w+1)] for __ in range(v+1)] # for already opened a gap in second string
    substitution_matrix = substitution_matrices.load("BLOSUM62") # load the substitution matrix
    gap_open_pen = penalty_params['gap_opening_penalty']
    gap_extend_pen = penalty_params.get('gap_extension_penalty', gap_open_pen)  # Equivalent to gap_extend_pen = gap_open_pen if it is linear

    # Fill dynamic programming tables with initial values
    for i in range(1, v+1):
        dp_2[i][0] = dp_1[i][0] = gap_open_pen
    for i in range(1, w+1):
        dp_2[0][i] = dp_1[0][i] = gap_open_pen
    dp_2[0][0] = dp_1[0][0] = gap_open_pen

    best_pos = (0, 0) # To hold maximum score and its position
    for i in range(1, v+1):
        for j in range(1, w+1):
            dp_1[i][j] = max(
                dp[i-1][j] + gap_open_pen,
                dp_1[i-1][j] + gap_extend_pen,
            )
            dp_2[i][j] = max(
                dp[i][j-1] + gap_open_pen,
                dp_2[i][j-1] + gap_extend_pen,
            )
            dp[i][j] = max(
                0,
                dp[i-1][j-1] + substitution_matrix[seq1[i-1]][seq2[j-1]],
                dp_1[i][j],
                dp_2[i][j]
            )
            
            # Update best position if necessary
            if(dp[i][j] > dp[best_pos[0]][best_pos[1]]):
                best_pos = (i, j)

    # Initialization for traceback
    i = best_pos[0]
    j = best_pos[1]
    alignment = ''
    alignment1 = seq1[i:]
    alignment2 = seq2[j:]
    plane = 0
    
    # Determine starting plane for the traceback
    if dp_1[i][j] == dp[i][j] and dp[i][j] != dp[i-1][j-1] + substitution_matrix[seq1[i-1]][seq2[j-1]]:
        plane = 1
    elif dp_2[i][j] == dp[i][j] and dp[i][j] != dp[i-1][j-1] + substitution_matrix[seq1[i-1]][seq2[j-1]]:
        plane = 2

    length = v+w+3 # To prevent infinite loop in case of a bug

    while(i > 0 or j > 0):
        assert length >= 0
        length -= 1

        # Stop when reached 0 value in the main dynamic programming matrix
        if dp[i][j] == 0 and plane == 0:
            alignment = ' ' * max(i, j) + alignment
            alignment1 = seq1[:i] + alignment1
            alignment2 = seq2[:j] + alignment2
            if i > j:
                alignment2 = ' ' * (i-j) + alignment2
            else:
                alignment1 = ' ' * (j-i) + alignment1
            break
        
        # If we are in the main matrix plane, find correct direction for traceback
        if plane == 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + substitution_matrix[seq1[i-1]][seq2[j-1]]:
                plane = 0
                alignment = ('|' if seq1[i-1] == seq2[j-1] else '.') + alignment
                alignment1 = seq1[i-1] + alignment1
                alignment2 = seq2[j-1] + alignment2
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + gap_open_pen:
                plane = 0
                alignment = '-' + alignment
                alignment1 = seq1[i-1] + alignment1
                alignment2 = '-' + alignment2
                i -= 1
            elif i > 0 and dp[i][j] == dp_1[i-1][j] + gap_extend_pen:
                plane = 1
                alignment = '-' + alignment
                alignment1 = seq1[i-1] + alignment1
                alignment2 = '-' + alignment2
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + gap_open_pen:
                plane = 0
                alignment = '-' + alignment
                alignment1 = '-' + alignment1
                alignment2 = seq2[j-1] + alignment2
                j -= 1
            else:
                plane = 2
                alignment = '-' + alignment
                alignment1 = '-' + alignment1
                alignment2 = seq2[j-1] + alignment2
                j -= 1
        # If we are in the first string plane, find correct direction for traceback
        elif plane == 1:
            if i > 0 and dp_1[i][j] == dp[i-1][j] + gap_open_pen:
                plane = 0
                alignment = '-' + alignment
                alignment1 = seq1[i-1] + alignment1
                alignment2 = '-' + alignment2
                i -= 1
            elif i > 0 and dp_1[i][j] == dp_1[i-1][j] + gap_extend_pen:
                plane = 1
                alignment = '-' + alignment
                alignment1 = seq1[i-1] + alignment1
                alignment2 = '-' + alignment2
                i -= 1
        # If we are in the second string plane, find correct direction for traceback
        else:
            if j > 0 and dp_2[i][j] == dp[i][j-1] + gap_open_pen:
                plane = 0
                alignment = '-' + alignment
                alignment1 = '-' + alignment1
                alignment2 = seq2[j-1] + alignment2
                j -= 1
            elif j > 0 and dp_2[i][j] == dp_2[i][j-1] + gap_extend_pen:
                plane = 2
                alignment = '-' + alignment
                alignment1 = '-' + alignment1
                alignment2 = seq2[j-1] + alignment2
                j -= 1 

    # Return score and alignment
    return dp[best_pos[0]][best_pos[1]], f'{alignment1}\n{alignment}\n{alignment2}\n'
