# Least Squares Fitting


import math

def evaluate_expression(expression, x, parameters):
    """
    Evaluate the expression for a given x and parameter values.
    """
    for i, param in enumerate(parameters):
        expression = expression.replace(f"a{i}", str(param))
    expression = expression.replace("x", str(x))
    return eval(expression)

def partial_derivative(expression, x, parameters, param_index):
    """
    Calculate the partial derivative of the expression with respect to a parameter.
    """
    h = 1e-6
    param_value = parameters[param_index]
    parameters[param_index] = param_value + h
    f_plus_h = evaluate_expression(expression, x, parameters)
    parameters[param_index] = param_value - h
    f_minus_h = evaluate_expression(expression, x, parameters)
    parameters[param_index] = param_value
    return (f_plus_h - f_minus_h) / (2 * h)

def least_squares_fit(expression, data_points, parameters, num_iterations=100, learning_rate=0.01):
    """
    Perform least squares fitting using gradient descent optimization.

    Gradient descent is an optimization algorithm used to minimize a cost function by
    iteratively adjusting the parameters in the direction of steepest descent.

    Reference:
    Ruder, S. (2016). An overview of gradient descent optimization algorithms.
    arXiv preprint arXiv:1609.04747.
    """
    for _ in range(num_iterations):
        gradients = [0] * len(parameters)
        for x, y in data_points:
            for i in range(len(parameters)):
                gradients[i] += (evaluate_expression(expression, x, parameters) - y) * \
                                partial_derivative(expression, x, parameters, i)
        for i in range(len(parameters)):
            parameters[i] -= learning_rate * gradients[i]
    return parameters

def main():
    data_points = []
    num_points = int(input("Enter the number of data points: "))
    print("Enter the data points (x, y):")
    for i in range(num_points):
        x, y = map(float, input(f"Data point {i+1}: ").split())
        data_points.append((x, y))

    expression = input("Enter the expression (e.g., a0 + a1*x + a2*x**2): ")
    num_params = expression.count("a")
    parameters = [float(input(f"Enter initial value for a{i}: ")) for i in range(num_params)]

    fitted_parameters = least_squares_fit(expression, data_points, parameters)

    print("\nFitted Parameters:")
    for i, param in enumerate(fitted_parameters):
        print(f"a{i} = {param:.4f}")

    print("\nFitted Expression:")
    print(expression.replace("a", "{:.4f}").format(*fitted_parameters))

if __name__ == "__main__":
    main()