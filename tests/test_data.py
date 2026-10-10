from src.data import prepare_data


def test_data_split_has_expected_size_and_positive_label():
    info = prepare_data(force=True)
    assert info["n_samples"] == 569
    assert info["n_features"] == 30
    assert info["n_train"] == 455
    assert info["n_test"] == 114
    import pandas as pd
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    assert set(train["target"].unique()) == {0, 1}
    assert set(test["target"].unique()) == {0, 1}
