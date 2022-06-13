dist_mat = pairwise_distance.copy() # copy the pairwise distance matrix
avg_distance = [0.0] * n # initialize the average distance
used = [False] * n # initialize the used array
newick_tree = [""] * n # initialize the newick trees
group_cnt = n # initialize the group count

while group_cnt >= 2:
    best_val = 1e9
    best_pair = -1, -1

    # Calculates Average Distance for each group
    for i in range(n):
        cur_dist = 0
        
        for j in range(n):
            if i == j or used[j]:
                continue
            cur_dist += dist_mat[i, j]

        if group_cnt > 2: # if there are 2 groups, there will be error
            avg_distance[i] = cur_dist / (group_cnt-2) # calculate the average distance for each group

    # Finds a pair that close each other but far from others
    for i in range(n):
        for j in range(n):
            if i == j or used[i] or used[j]:
                continue
            if best_val > dist_mat[i, j] - avg_distance[i] - avg_distance[j]:
                best_val = dist_mat[i, j] - avg_distance[i] - avg_distance[j]
                best_pair = i, j
    
    g1, g2 = best_pair # Always g1 < g2, g1 and g2 are the two groups to merge
    
    # Calculate branch lengths
    first_branch_length = (dist_mat[g1, g2] + (avg_distance[g1] - avg_distance[g2])) / 2.0
    second_branch_length = (dist_mat[g1, g2] + (avg_distance[g2] - avg_distance[g1])) / 2.0
    
    if group_cnt == 2:
        # If there is only 2 groups, then the first branch is the only branch
        # g1 will be root of the tree, therefore g2 will be the child of g1
        # first branch length will be 0
        # second branch length will be the distance between g1 and g2
        first_branch_length = 0
        second_branch_length = dist_mat[g1, g2]
    
    used[g2] = True # Mark the second group as used
    group_cnt -= 1 # Decrement the group count

    # Update the distance matrix
    # g1 will be used as the new node
    for i in range(n):
        if i == g1 or used[i]:
            continue
        
        dist_mat[i, g1] = (dist_mat[g1, i] + dist_mat[g2, i] - dist_mat[g1, g2]) / 2.0 # Update the distance matrix
        dist_mat[g1, i] = dist_mat[i, g1] # Symmetric matrix
    
    # Update newick tree string
    # g1 will be always the new node
    if newick_tree[g1] == "" and newick_tree[g2] == "":
        newick_tree[g1] = "(" + str(g1) + ":" + str(first_branch_length) + "," + str(g2) + ":" + str(second_branch_length) + ")" 
    elif newick_tree[g2] == "":
        newick_tree[g1] = "(" + newick_tree[g1] + ":" + str(first_branch_length) + "," + str(g2) + ":" + str(second_branch_length) + ")"
    elif newick_tree[g1] == "":
        newick_tree[g1] = "(" + str(g1) + ":" + str(first_branch_length) + "," + newick_tree[g2] + ":" + str(second_branch_length) + ")"
    else:
        newick_tree[g1] = "(" + newick_tree[g1] + ":" + str(first_branch_length) + "," + newick_tree[g2] + ":" + str(second_branch_length) + ")"

newick_tree[0]