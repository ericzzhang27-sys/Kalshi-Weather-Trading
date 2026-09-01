import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from src.kalshi.normalize_markets import parse_kalshi_bucket
from src.bucket_schema import TemperatureBucket

def test_less_bucket_from_strikes():
    m = {"ticker":"KXHIGHNY-TEST1","floor_strike":None,"cap_strike":82,"strike_type":"less","yes_sub_title":"82 or below"}
    b = parse_kalshi_bucket(m)
    assert b.lower_temp is None
    assert b.upper_temp == 81.5

def test_greater_bucket_from_strikes():
    m = {"ticker":"KXHIGHNY-TEST2","floor_strike":87,"cap_strike":None,"strike_type":"greater","yes_sub_title":"87 or above"}
    b = parse_kalshi_bucket(m)
    assert b.lower_temp == 87.5
    assert b.upper_temp is None

def test_between_bucket():
    m = {"ticker":"KXHIGHNY-TEST3","floor_strike":83,"cap_strike":84,"strike_type":"between","yes_sub_title":"83 to 84"}
    b = parse_kalshi_bucket(m)
    assert b.lower_temp == 82.5
    assert b.upper_temp == 84.5


def test_legacy_equal_text_bounds_are_one_degree_bucket():
    b = parse_kalshi_bucket({
        "ticker": "HIGHNY-22JUL23-B96.0",
        "subtitle": "96-96°",
        "rules_primary": "temperature is between 96-96°",
    })
    assert b.lower_temp == 95.5
    assert b.upper_temp == 96.5

def test_text_fallback_less():
    m = {"ticker":"KXHIGHNY-TEST4","title":"Will high temp be < 85?","subtitle":"84 or below"}
    b = parse_kalshi_bucket(m)
    # Should infer less via text fallback
    assert b.upper_temp is not None

def test_bucket_contiguity_validation():
    # Create 6-bucket NYC style around 73
    from src.bucket_schema import validate_temperature_buckets
    buckets = [
        TemperatureBucket(label="69 or lower", lower_temp=None, upper_temp=69.5),
        TemperatureBucket(label="70 to 71", lower_temp=69.5, upper_temp=71.5),
        TemperatureBucket(label="72 to 73", lower_temp=71.5, upper_temp=73.5),
        TemperatureBucket(label="74 to 75", lower_temp=73.5, upper_temp=75.5),
        TemperatureBucket(label="76 to 77", lower_temp=75.5, upper_temp=77.5),
        TemperatureBucket(label="78 or higher", lower_temp=77.5, upper_temp=None),
    ]
    validate_temperature_buckets(buckets)

def test_duplicate_bucket_detection():
    from src.trading.contract_mapping import validate_contract_mapping
    buckets = [
        TemperatureBucket(label="70 to 71", lower_temp=69.5, upper_temp=71.5),
        TemperatureBucket(label="70 to 71 dup", lower_temp=69.5, upper_temp=71.5),
    ]
    v = validate_contract_mapping(buckets)
    assert not v.valid
    assert "duplicate" in v.no_trade_reason
