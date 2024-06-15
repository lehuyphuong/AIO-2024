import gdown


def word_in_text_count(content):
    """
    This function is created to count how many
    each words present in content
    """
    symbol = [".", ";", "?", "_", "!", "@", "#",
              "(", ")", "[", "]", "-", "%", "*",
              ","]
    unique_words = []
    text_words = str(content.split('\n'))
    # print(text_lines)
    text_words = text_words.split(' ')

    # Convert Upper-case words into Low-case ones
    for index in range(len(text_words)):
        text_words[index] = text_words[index].lower()

    new_word = ''
    new_text_words = []

    # Remove symbols
    for word in text_words:
        new_word = word
        for letter in new_word:
            if letter in symbol:
                new_word = word.replace(letter, '')
        new_text_words.append(new_word)

    # Collect all unique words
    for word in new_text_words:
        if word not in unique_words:
            unique_words.append(word)

    # Count number of present in each unique words
    for unique_word in unique_words:
        print("this word '{}' displays {} times".format(
            unique_word,
            new_text_words.count(unique_word)))


if __name__ == "__main__":
    input_user = ''
    file_path = 'P1_data.txt'
    gdown.download(
        "https://drive.google.com/uc?id=1IBScGdW2xlNsc9v5zSAya548kNgiOrko",
        file_path,
        quiet=False)

    while True:
        try:
            input_user = file_path
            with open(input_user, 'r') as var_read_lines:
                text_content = var_read_lines.read()
                word_in_text_count(text_content)

            break
        except ValueError:
            print("file does not exit, try again!")
