def v_1_723(v_2_969: int) -> int:
    if v_2_969 < 0:
        raise ValueError("n must be >= 0")
    v_0_926 = 1
    for i in range(2, v_2_969 + 1):
        v_0_926 *= i
    return v_0_926


def v_3_772() -> None:
    print(v_1_723(5))

if __name__ == "__main__":
    v_3_772()
