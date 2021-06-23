import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# credit: https://heartbeat.fritz.ai/introduction-to-deep-learning-with-keras-c7c3d14e1527

df = pd.read_csv('../data/insurance_claims.csv')
sc = StandardScaler()

params = {
    'batch_size': [20, 35],
    'nb_epoch': [150, 500],
    'optimizer': ['adam', 'rmsprop']
}

# convert the categorical columns to dummy variables
feats = ['policy_state', 'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies',
         'insured_relationship', 'collision_type', 'incident_severity', 'authorities_contacted', 'incident_state',
         'incident_city', 'incident_location', 'property_damage', 'police_report_available', 'auto_make', 'auto_model',
         'fraud_reported', 'incident_type']
df_final = pd.get_dummies(df, columns=feats, drop_first=True)  # drop_first=True to avoid the dummy variable trap

# drop our prediction value column and other non essentials
X = df_final.drop(['fraud_reported_Y', 'policy_csl', 'policy_bind_date', 'incident_date'],
                  axis=1).values  # .values returns numpy arrays
y = df_final['fraud_reported_Y'].values

# split the data into training and test (70-30) respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# scale data
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def make_classifier(optimizer):
    # initialize ANN
    classifier = Sequential()

    # add creates first hidden layer using Dense
    # add 3 nodes in first layer, initialize weights uniform close to 0, use ReLu activation function, input layer is 5 nodes
    classifier.add(
        Dense(3, kernel_initializer='uniform',
              activation='relu', input_dim=1143))

    # add second hidden layer
    classifier.add(
        Dense(3, kernel_initializer='uniform',
              activation='relu'))

    # add output layer
    classifier.add(
        Dense(1, kernel_initializer='uniform',
              activation='sigmoid'))

    # compile ANN using optimizer var for parameter tuning
    # our problem is binary so use binary_crossentropy
    # criterion to evaluate is accuracy
    classifier.compile(optimizer=optimizer,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    return classifier


classifier = KerasClassifier(build_fn=make_classifier)

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=params,
                           scoring='accuracy',
                           cv=10)

grid_search = grid_search.fit(X_train, y_train)

best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_param)
print(best_accuracy)
