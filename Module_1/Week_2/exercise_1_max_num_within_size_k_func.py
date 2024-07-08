def find_max_within_sliding_win(string, k):
    """
    This function is created to find maximum number within k elements
    . Also, the k will slide from left to right in string
    """
    list_all_max_nums = []
    string_of_k = []
    index_string = 0
    for index_string in range(0, len(string)-2):
        for _ in range(0, k):
            string_of_k.append(string[index_string])
            index_string += 1

        list_all_max_nums.append(max(string_of_k))
        string_of_k.clear()

    return list_all_max_nums


if __name__ == "__main__":
    input_user_string = []
    input_user_k = 0
    num_of_elements = 0
    while True:
        try:
            input_user_string = input("Pls type your string number: ")
            input_user_string = input_user_string.split()
            num_of_elements = len(input_user_string)
            for index in range(num_of_elements):
                input_user_string[index] = float(input_user_string[index])
            break

        except ValueError:
            print("The string is not a list of numbers, try again")
            print("Hint: use blank space to sperate element")

    while True:
        try:
            input_user_k = int(input("Please type sliding window k: "))
            if input_user_k <= num_of_elements:
                print(find_max_within_sliding_win(
                    input_user_string,
                    input_user_k))
                break
            else:
                print("sliding window is bigger than the string, try again")

        except ValueError:
            print("sliding window must be an integer number, pls try again!")
