from timeit import timeit

cy = timeit('test_cy.looper(50000)', setup='import test_cy', number=10000)
py = timeit('test_py.looper(50000)', setup='import test_py', number=10000)

print("Cy is on average {}x faster than py".format(py/cy))
