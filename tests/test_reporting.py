import pandas as pd

from src.reporting import markdown_table, objective_verdict, rupees


def test_rupees_uses_indian_digit_grouping():
    assert rupees(504127) == "Rs 5,04,127"
    assert rupees(8900000) == "Rs 89,00,000"
    assert rupees(999) == "Rs 999"
    assert rupees(-1500) == "Rs -1,500"


def test_markdown_table_has_a_header_and_a_divider():
    frame = pd.DataFrame({"Model": ["XGBoost"], "R2": [0.7481234]})
    rendered = markdown_table(frame).splitlines()
    assert rendered[0] == "| Model | R2 |"
    assert rendered[1] == "| --- | --- |"
    assert "0.7481" in rendered[2]


def test_objective_verdict_does_not_round_a_miss_into_a_pass():
    assert objective_verdict(0.93)[0] == "MET"
    assert objective_verdict(0.8999)[0] == "NOT MET"
    assert objective_verdict(0.90)[0] == "MET"
