from uncert_ident.utilities import time_decorator


@time_decorator
def looper(x):
    y = 0
    for i in range(x):
        y += i
    return y


looper(100)