def factorial(num: int) -> int:
    if num < 0:
        raise ValueError("n must be >= 0")
    out = 1
    for i in range(2, num + 1):
        out *= i
    return out


def main() -> None:
    print(factorial(5))


if __name__ == "__main__":
    main()
