import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv('gpascore.csv')

# pre-processing
# print(data.isnull().sum())
data = data.dropna() #remove NaN/blank rows
# data = data.fillna(100)

# print(data['gpa'].min())
# print(data['gpa'].count())

y_data = data['admit'].values

x_data = []
for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])


# 1. design deep learning model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    # last layer with 1 ouput, sigmoid allows output range between 0 and 1
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# 2. Compile model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Fit model
model.fit(np.array(x_data), np.array(y_data), epochs=1000)

# 4. Prediction
predict = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(predict)

# data preprocessing and hyperparameter tuning can increase accuracy

def edit_distance(x, y):
    """
    args:
        x:string = the first word.
        y:string = The second word.
    
    return:
        Tuple[String,String] = the optimum global alignment between x and y. The first string in the 
        tuple corresponds to x and the second to y. Use hypen's '-' to represent gaps in each string.
    """
    n = len(x)
    m = len(y)
    costs = [[0] * (m+1) for k in range(n+1)]
    
    x_opt_align = ""
    y_opt_align = ""
    
    for i in range(n+1):
        costs[i][0] = i
        
    for j in range(m+1):
        costs[0][j] = j
        
    for i in range(1, n+1):
        for j in range(1, m+1):
            if x[i-1] == y[j-1]:
                costs[i][j] = costs[i-1][j-1]
            else: 
                costs[i][j] = min(costs[i-1][j], costs[i][j-1], costs[i-1][j-1]) + 1
            
    i = n
    j = m
    while i > 0 or j > 0: 
        if i > 0 and j > 0 and x[i-1] == y[j-1]:
            x_opt_align = x[i-1] + x_opt_align
            y_opt_align = y[j-1] + y_opt_align
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and costs[i][j] == 1 + costs[i-1][j-1]:
            x_opt_align = x[i-1] + x_opt_align
            y_opt_align = y[j-1] + y_opt_align
            i -= 1
            j -= 1
        elif i > 0 and costs[i][j] == 1 + costs[i-1][j]:
            x_opt_align = x[i-1] + x_opt_align
            y_opt_align = '-' + y_opt_align
            i -= 1
        elif j > 0 and costs[i][j] == 1 + costs[i][j-1]:
            x_opt_align = '-' + x_opt_align
            y_opt_align = y[j-1] + y_opt_align
            j -= 1
    
    return (x_opt_align, y_opt_align)