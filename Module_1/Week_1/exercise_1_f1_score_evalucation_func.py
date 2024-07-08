def f1_score_evaluation(tp, fp, fn):
    # Calculate precision
    precision = tp / (tp + fp)

    # Calculate recall
    recall = tp / (tp + fn)

    # Calculate f1 score
    f1_score = (2 * (precision * recall)) / (precision + recall)

    return precision, recall, f1_score


def check_valid_input(prompt):
    while True:
        try:
            input_user = int(input(prompt))
            if input_user > 0:
                return input_user
            else:
                print("This input must be greater than 0, please try again!")
        except ValueError:
            print(" This input must be integer, please try again!")


if __name__ == "__main__":
    tp = check_valid_input("Please enter tp value: ")
    fp = check_valid_input("Please enter fp value: ")
    fn = check_valid_input("Please enter fn value: ")

    precision, recall, f1_score = f1_score_evaluation(tp, fp, fn)

    print("precision value = {:.5f}".format(precision))
    print("recall value = {:.5f}".format(recall))
    print("f1_score value = {:.5f}".format(f1_score))
