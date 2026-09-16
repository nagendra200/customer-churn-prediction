from src.data import add_cltv, generate_customers


def test_generated_data_is_reproducible() -> None:
    first = generate_customers(rows=100, seed=7)
    second = generate_customers(rows=100, seed=7)
    assert first.equals(second)


def test_cltv_is_positive() -> None:
    frame = add_cltv(generate_customers(rows=100))
    assert (frame["cltv"] > 0).all()
