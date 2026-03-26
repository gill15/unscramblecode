def compute_factorial(count: int) -> int:
    if count < 0:
        raise ValueError("n must be >= 0")
    out = 1
    for i in range(2, count + 1):
        out *= i
    return out


def main() -> None:
    print(compute_factorial(5))


if __name__ == "__main__":
    main()
