import numpy as np


def create_train_data():
    data = [['Sunny',   'Hot',  'High',     'Weak',     'no'],
            ['Sunny',   'Hot',  'High',     'Strong',   'no'],
            ['Overcast', 'Hot',  'High',     'Weak',     'yes'],
            ['Rain',    'Mild', 'High',     'Weak',     'yes'],
            ['Rain',    'Cool', 'Normal',   'Weak',     'yes'],
            ['Rain',    'Cool', 'Normal',   'Strong',   'no'],
            ['Overcast', 'Cool', 'Normal',   'Strong',   'yes'],
            ['Overcast', 'Mild', 'High',     'Weak',     'no'],
            ['Sunny',   'Cool', 'Normal',   'Weak',     'yes'],
            ['Rain',    'Mild', 'Normal',   'Weak',     'yes']]
    return np.array(data)


train_data = create_train_data()
print(train_data)


def compute_prior_probability(train_data):
    depth = np.shape(train_data)
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))
    unique, counts = np.unique(train_data, return_counts=True)
    package = dict(zip(unique, counts))

    prior_probability[0] = package['no']/depth[0]
    prior_probability[1] = package['yes']/depth[0]

    return prior_probability


prior_probability = compute_prior_probability(train_data=train_data)
print("P(play tennis = no) = ", prior_probability[0])
print("P(play tennis = yes) = ", prior_probability[1])


def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []

    for i in range(0, train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

        x_conditional_probability = []
        for y in y_unique:
            y_count = np.sum(train_data[:, -1] == y)
            if y_count == 0:
                x_conditional_probability.append(
                    [0 for _ in range(len(x_unique))])
                continue
            y_probs = []
            for x in x_unique:
                count = np.sum((train_data[:, i] == x)
                               & (train_data[:, -1] == y))
                y_probs.append(count / y_count)
            x_conditional_probability.append(y_probs)

        conditional_probability.append(x_conditional_probability)

    return conditional_probability, list_x_name


train_data = create_train_data()
_, list_x_name = compute_conditional_probability(train_data)
print("x1 = ", list_x_name[0])
print("x2 = ", list_x_name[1])
print("x3 = ", list_x_name[2])
print("x4 = ", list_x_name[3])


def get_index_from_value(feature_name, list_features):
    return np.nonzero(list_features == feature_name)[0][0]


train_data = create_train_data()
_, list_x_name = compute_conditional_probability(train_data=train_data)
outlook = list_x_name[0]

i1 = get_index_from_value("Overcast", outlook)
i2 = get_index_from_value("Rain", outlook)
i3 = get_index_from_value("Sunny", outlook)

print(i1, i2, i3)


train_data = create_train_data()
conditional_probability, list_x_name = compute_conditional_probability(
    train_data)
# Compute P(" Outlook "=" Sunny "| Play Tennis "=" Yes ")
x1 = get_index_from_value("Sunny", list_x_name[0])
print("P(Outlook=Sunny|Play Tennis=Yes)=",
      np.round(conditional_probability[0][1][x1], 2))


train_data = create_train_data()
conditional_probability, list_x_name = compute_conditional_probability(
    train_data)
# Compute P(" Outlook "=" Sunny "| Play Tennis "=" No ")
x1 = get_index_from_value("Sunny", list_x_name[0])
print("P(Outlook=Sunny|Play Tennis=No)=", np.round(conditional_probability
                                                   [0][0][x1], 2))

# Train Naive Bayes Model


def compute_prior_probablity(train_data):
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))
    for i in range(0, len(y_unique)):
        prior_probability[i] = len(
            np.nonzero(train_data[:, 4] == y_unique[i])[0])/len(train_data)
    return prior_probability


def train_naive_bayes(train_data):
    # Step 1: Calculate Prior Probability
    prior_probability = compute_prior_probablity(train_data)

    # Step 2: Calculate Conditional Probability
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    return prior_probability, conditional_probability, list_x_name


def prediction_play_tennis(event_x, list_x_name, prior_probability, conditional_probability):

    x1 = get_index_from_value(event_x[0], list_x_name[0])
    x2 = get_index_from_value(event_x[1], list_x_name[1])
    x3 = get_index_from_value(event_x[2], list_x_name[2])
    x4 = get_index_from_value(event_x[3], list_x_name[3])

    p0 = prior_probability[0] * \
        conditional_probability[0][0][x1] * \
        conditional_probability[1][0][x2] * \
        conditional_probability[2][0][x3] * \
        conditional_probability[3][0][x4]

    p1 = prior_probability[1] * \
        conditional_probability[0][1][x1] * \
        conditional_probability[1][1][x2] * \
        conditional_probability[2][1][x3] * \
        conditional_probability[3][1][x4]

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


X = ['Sunny', 'Cool', 'High', 'Strong']
data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(
    data)
pred = prediction_play_tennis(
    X, list_x_name, prior_probability, conditional_probability)

if (pred):
    print("Ad should go!")
else:
    print("Ad should not go!")
