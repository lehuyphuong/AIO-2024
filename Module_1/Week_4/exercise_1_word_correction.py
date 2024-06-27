import streamlit as st


def load_vocab(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    words = sorted(set([line.strip().lower() for line in lines]))
    return words


def cal_dis_levenshtein(token1, token2):
    """
    This function is created for calculating distance that transforming
    a source to target
    """
    src_len = len(token1)
    tar_len = len(token2)

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
            if token1[i - 1] == token2[j - 1]:
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
    st.title("Word Correction using Levenshtein Distance")
    word = st.text_input('Word:')
    vocabs = load_vocab(
        file_path='vocab.txt')

    if st.button("Compute"):

        # compute Levenshtein Distance
        leven_distances = dict()
        for vocab in vocabs:
            leven_distances[vocab] = cal_dis_levenshtein(word, vocab)

        # sorted by distance
        sorted_distances = dict(
            sorted(leven_distances.items(), key=lambda item: item[1]))
        correct_word = list(sorted_distances.keys())[0]
        st.write('Correct word: ', correct_word)

        col1, col2 = st.columns(2)
        col1.write('Vocabulary')
        col1.write(vocabs)

        col2.write('Distance')
        col2.write(sorted_distances)
