

# fit function form
def func(xy, a, b, c, d, e, f): 
    x, y = xy 
    
    print('~~~~~~~')
    print('fit function of form a + bx + cy + dx^2 + ey^2 + fxy')
    print('~~~~~~~')
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y
