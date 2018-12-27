import numpy
import theano.tensor as T
import sys
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x+y
x1 = float(sys.argv[1])
y1 = float(sys.argv[2])
f = function([x, y], z)

z1 = f(x1,y1)
print(z1)