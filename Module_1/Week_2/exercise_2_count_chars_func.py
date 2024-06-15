def count_chars(list_of_characters):
    """
    This function is created for counting number of presents
    in unique character
    """
    blank_space_request = ''
    blank_space_request_flag = False
    list_of_unique_chars = []

    for letter in list_of_characters:
        if letter not in list_of_unique_chars:
            list_of_unique_chars.append(letter)

    if ' ' in list_of_unique_chars:
        while True:
            try:
                blank_space_request = input(
                    "Blank space is found, would you like to count it(Y/N)? ")
                if blank_space_request == "Y":
                    blank_space_request_flag = True
                    break
                elif blank_space_request == "N":
                    blank_space_request_flag = False
                    break
                else:
                    print("Only Y or N, pick one!")

            except ValueError:
                print("Only Y or N, pick one!")

    if blank_space_request_flag is True:
        for letter in list_of_unique_chars:
            print("this {} displays {} times".format(
                letter,
                list_of_characters.count(letter)))
    else:
        list_of_unique_chars.remove(' ')
        for letter in list_of_unique_chars:
            print("this {} displays {} times".format(
                letter,
                list_of_characters.count(letter)))


if __name__ == "__main__":
    input_user_string = []
    list_of_characters = []
    while True:
        try:
            input_user_string = input("Pls type your string: ")
            for letter in input_user_string:
                list_of_characters.append(letter)

            count_chars(list(list_of_characters))
            break

        except ValueError:
            print("The string is not a list of numbers, try again")
            print("Hint: use blank space to sperate element")
