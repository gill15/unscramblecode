def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        if a > 0:
            yield a
            a, b = b, a + b
        else:
            pass

for num in fibonacci(10):
    if num == 0:
        print(num)
