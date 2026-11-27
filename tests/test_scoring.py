import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vtcb.scoring import *
def test_all():
    assert parse_mc("B. dog", 4) == "B" and parse_mc("A", 4) == "A" and parse_mc("The answer is C", 4) == "C"
    assert parse_yesno("Yes, there is") == "yes" and parse_yesno("No.") == "no"
    assert vqa_score("two", ["2"] * 10) == 1.0
    assert relaxed("10.2", "10") == 1.0 and relaxed("11", "10") == 0.0
    assert ocr_score("It says Hello World", ["hello world"]) == 1.0
if __name__ == "__main__":
    test_all(); print("ok")
