import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import regularizers
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

data = pd.read_csv("network_id.csv")

print(data.head())

plt.style.use('tableau-colorblind10')

def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    for ax, col in zip(axes.ravel(), cols_list):
        counts = df[col].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.0f%%', startangle=90, textprops={'fontsize': 15})
        ax.set_title(col, fontsize=15)
        ax.axis('equal')
    plt.tight_layout()
    plt.show()

pie_plot(data, ['protocol_type', 'class'], 1, 2)

def do_scl(df_num, cols):
    print("Original values:\n", df_num)

    scaler = RobustScaler()
    scaler_temp = scaler.fit_transform(df_num)
    std_df = pd.DataFrame(scaler_temp, columns =cols)

    print("\nScaled values:\n", std_df)

    return std_df

cat_cols = ['protocol_type','service','flag', 'class']

def process(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = do_scl(df_num, num_cols)

    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]

    dataframe.loc[dataframe['class'] == "normal", "class"] = 0
    dataframe.loc[dataframe['class'] != 0, "class"] = 1

    print("Before encoding:")
    print(dataframe['protocol_type'])

    dataframe = pd.get_dummies(dataframe, columns = ['protocol_type', 'service', 'flag'])

    print("\nColumns after encoding:")
    print(dataframe.filter(regex='^protocol_type_'))
    
    return dataframe

scaled_train = process(data)

x = scaled_train.drop(['class'] , axis = 1).values
pca = PCA(n_components=20)
pca = pca.fit(x)
x_reduced = pca.transform(x)

y = scaled_train['class'].values
y = y.astype('int')

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=42)
x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = \
    train_test_split(x_reduced, y, test_size=0.2, random_state=42)

kernal_evals = dict()

def evaluation(model, name, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Time " + str(name) + " : %.3f sec" % execution_time)

    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

    train_precision = metrics.precision_score(y_train, model.predict(X_train))
    test_precision = metrics.precision_score(y_test, model.predict(X_test))

    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    test_recall = metrics.recall_score(y_test, model.predict(X_test))

    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    print("Training Accuracy " + str(name) + " {}  Test Accuracy ".format(train_accuracy*100) + str(name) + " {}".format(test_accuracy*100))
    print("Training Precision " + str(name) + " {}  Test Precision ".format(train_precision*100) + str(name) + " {}".format(test_precision*100))
    print("Training Recall " + str(name) + " {}  Test Recall ".format(train_recall*100) + str(name) + " {}".format(test_recall*100))

    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay( \
        confusion_matrix =                       \
        confusion_matrix,                        \
        display_labels = ['normal', 'attack'])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(False)
    ax.set_title(name, fontsize=15)
    cm_display.plot(ax=ax)
    plt.show()

lr = LogisticRegression().fit(x_train, y_train)
evaluation(lr, "Logistic Regression", x_train, x_test, y_train, y_test)

knn = KNeighborsClassifier(n_neighbors=20).fit(x_train, y_train)
evaluation(knn, "KNeighborsClassifier", x_train, x_test, y_train, y_test)

gnb = GaussianNB().fit(x_train, y_train)
evaluation(gnb, "GaussianNB", x_train, x_test, y_train, y_test)

xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
xgb_classifier.fit(x_train, y_train)

evaluation(xgb_classifier, "XGBoost", x_train, x_test, y_train, y_test)

rf = RandomForestClassifier().fit(x_train, y_train)
evaluation(rf, "RandomForestClassifier", x_train, x_test, y_train, y_test)

rrf = RandomForestClassifier().fit(x_train_reduced, y_train_reduced)
evaluation(rrf, "PCA RandomForest", x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(x_train.shape[1:]),
                 kernel_regularizer=regularizers.l2(0.01),
                 bias_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=32, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01),
                 bias_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=1)

train_predictions = model.predict(x_train)

start_time = time.time()
test_predictions = model.predict(x_test)
end_time = time.time()
execution_time = end_time - start_time
print("Time Neural network: %.3f sec" % execution_time)

train_predictions_binary = np.round(train_predictions)
test_predictions_binary = np.round(test_predictions)

train_accuracy = metrics.accuracy_score(y_train, train_predictions_binary)
train_precision = metrics.precision_score(y_train, train_predictions_binary)
train_recall = metrics.recall_score(y_train, train_predictions_binary)

test_accuracy = metrics.accuracy_score(y_test, test_predictions_binary)
test_precision = metrics.precision_score(y_test, test_predictions_binary)
test_recall = metrics.recall_score(y_test, test_predictions_binary)

print("Training Set Metrics:")
print("Accuracy: {:.2f}".format(train_accuracy))
print("Precision: {:.2f}".format(train_precision))
print("Recall: {:.2f}".format(train_recall))

print("\nTest Set Metrics:")
print("Accuracy: {:.2f}".format(test_accuracy))
print("Precision: {:.2f}".format(test_precision))
print("Recall: {:.2f}".format(test_recall))