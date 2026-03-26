import v_4_825

def sum_amounts(v_7_986: str) -> float:
    v_0_710 = 0.0
    with open(v_7_986, newline="") as f:
        v_5_330 = v_4_825.DictReader(f)
 for v_6_536 in v_5_330:
# v_3_31: v_2_234 v_1_172
 v_0_710 += float(v_6_536["amount"])
    return v_0_710

# v_3_31: v_2_234 v_1_172
