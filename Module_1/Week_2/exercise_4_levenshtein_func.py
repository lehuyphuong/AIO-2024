def cal_dis_levenshtein(src, tar):
    """
    This function is created for calculating distance that transforming
    a source to target
    """
    src_len = len(src)
    tar_len = len(tar)

    # Step 1: Initialize distance matrix to store value
    dis_mat = [[0 for _ in range(tar_len + 1)] for _ in range(src_len + 1)]

    # Step 2: Fill index for first row: source and column: target
    for i in range(src_len + 1):
        dis_mat[i][0] = i

    for j in range(tar_len + 1):
        dis_mat[0][j] = j

    # Step 3: Fill the rest matrix
    for i in range(1, src_len + 1):
        for j in range(1, tar_len + 1):
            # If the letter between is matched, no need to measure
            if src[i - 1] == tar[j - 1]:
                dis_mat[i][j] = dis_mat[i - 1][j - 1]
            else:
                dis_mat[i][j] = 1 + min(
                    dis_mat[i][j - 1],  # Insert
                    dis_mat[i - 1][j],  # Delete
                    dis_mat[i - 1][j-1]  # ReplaceS
                )
    # Step 4: Find distance
    return dis_mat[src_len][tar_len]


if __name__ == "__main__":
    source_string = input("Put source here: ")
    target_string = input("Put target here: ")

    distance = cal_dis_levenshtein(source_string, target_string)

    print("The distance that convert from '{}' to '{}' is {}"
          .format(source_string, target_string, distance))
