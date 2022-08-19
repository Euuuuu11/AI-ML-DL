param_bounds = {'x1' : (-1, 5),
                'x2' : (0, 4)}
def y_function(x1, x2):
    return -x1 **2 - (x2 - 2) **2 + 10

# pip install bayesian-optimization
from pickletools import optimize
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(f=y_function,
                                 pbounds=param_bounds,
                                 random_state=1234)     #   f찾고자 하는 함수

optimizer.maximize(init_points=2,
                   n_iter=20)

print(optimizer.max)    
# {'target': 9.999835918969607, 'params': {'x1': 0.00783279093916099, 'x2': 1.9898644972252864}}