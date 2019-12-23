from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

def best_slope(xs , ys) :
    up = ( (mean(xs)*mean(ys)) - mean(xs*ys) )
    down = ( (mean(xs)**2) - mean(xs**2) )
    m = up / down
    return m

def best_intercept(xs,ys,m):
    b = ( mean(ys) - (m*mean(xs)) )
    return b

def squared_error(ys,line):
    return sum( (line - ys) ** 2 )

def coefficient_of_determinent(ys,line):
    mean_line = [mean(ys) for y in ys]
    reg_line_sq_error = squared_error(ys,line)
    mean_line_sq_error = squared_error(ys,mean_line)
    return ( 1 - ( reg_line_sq_error / mean_line_sq_error ) )


# sample dataset
xs = [1,2,3,4,5,6,7,8,9,10]
ys = [1,2,5,7,9,11,11,15,18,19]

xs = np.array(xs, dtype=np.float64)
ys = np.array(ys, dtype=np.float64)

# making best fit line 
m = best_slope(xs , ys)
b = best_intercept(xs, ys, m)
regression_line = [ m*x + b for x in xs ]

plt.plot(regression_line)
plt.scatter(xs,ys)

# testing accuracy
r_squared = coefficient_of_determinent(ys, regression_line)
print('R_Squared is : ' , r_squared)

if r_squared > .6:
    print('Dataset is suitable for linear regression')
else:
    print('Dataset is not suitable for linear regression')

# target dataset
predict_x = 15
predict_y = m*predict_x + b
print(predict_y)

plt.scatter(predict_x, predict_y)
plt.show()