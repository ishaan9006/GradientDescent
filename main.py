import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cost_function(y, y_predicted):
    curr_cost = sum((y_predicted - y) ** 2) / len(y)
    return curr_cost

def gradient_descent(X, Y, initial_weight, initial_bias, epoch = 1000, learning_rate = 0.0001, stopping_threshold = 1e-6):
    learning_rate = learning_rate
    
    bias = initial_bias
    weight = initial_weight

    prev_cost = None
    # Array for storing each cost and weight value
    costs = []
    weights = []
    # Calculating the current cost
    m = float(len(X))

    # Loop running for the given iterations
    for i in range(0, epoch):
        y_predicted = (weight * X) + bias
        
        cost = cost_function(Y, y_predicted)

        if prev_cost and abs(prev_cost - cost) <= stopping_threshold:
            break
        prev_cost = cost

        costs.append(cost)
        weights.append(weight)


        weight -= (learning_rate / m) * sum((y_predicted - Y) * X) 
        bias -= (learning_rate / m) * sum(y_predicted - Y)

        print(f"For iteration {i+1} Weight: {weight}  Bias: {bias} Cost {cost} ")

    return weight, bias

def run():
    # Data
    data = pd.read_csv('data.csv')
    X = np.array(data['X'].values)
    Y = np.array(data['Y'].values)
    
    # Assumptions for the model
    epoch = 1000
    learning_rate = 0.001
    initial_weight = 0
    initial_bias = 0

    weight, bias = gradient_descent(X, Y, initial_weight, initial_bias)
    print("\n")
    print(f"Final Values for Weight: {weight} and Bias: {bias}")

    Y_pred = weight*X + bias
 
    # Plotting the regression line
    plt.figure(figsize = (8,6))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red', markersize=10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    

if __name__=="__main__":
    run()
 