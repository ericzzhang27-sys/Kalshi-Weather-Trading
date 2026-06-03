from __future__ import annotations

import pandas as pd
import pytest

from src.trading.contract_mapping import (
    ContractMappingError,
    map_event_contracts,
    parse_contract_bucket,
    validate_contract_mapping,
)


def _market(**overrides):
    base = {
        "event_ticker": "KXHIGHNY-26JUN02",
        "ticker": "KXHIGHNY-26JUN02-B75.5",
        "status": "active",
        "eligible": True,
        "title": "Will the high temp in NYC be 75-76 on Jun 2, 2026?",
        "subtitle": "75 to 76",
        "rules_primary": "If the highest temperature is between 75-76, then Yes.",
        "strike_type": "between",
        "floor_strike": 75,
        "cap_strike": 76,
    }
    base.update(overrides)
    return base


def test_parse_lower_tail_contract_from_cap_strike() -> None:
    bucket = parse_contract_bucket(
        _market(
            ticker="KXHIGHNY-26JUN02-T73",
            title="Will the high temp in NYC be <73 on Jun 2, 2026?",
            subtitle="72 or below",
            strike_type="less",
            floor_strike=None,
            cap_strike=73,
        )
    )

    assert bucket.lower_temp is None
    assert bucket.upper_temp == 72.5


def test_parse_upper_tail_contract_from_floor_strike() -> None:
    bucket = parse_contract_bucket(
        _market(
            ticker="KXHIGHNY-26JUN02-T80",
            title="Will the high temp in NYC be >80 on Jun 2, 2026?",
            subtitle="81 or above",
            strike_type="greater",
            floor_strike=80,
            cap_strike=None,
        )
    )

    assert bucket.lower_temp == 80.5
    assert bucket.upper_temp is None


def test_parse_between_contract_from_integer_strikes() -> None:
    bucket = parse_contract_bucket(_market())

    assert bucket.lower_temp == 74.5
    assert bucket.upper_temp == 76.5


def test_missing_required_strike_fields_rejects_contract() -> None:
    with pytest.raises(ContractMappingError, match="missing_strikes"):
        parse_contract_bucket(
            _market(
                title="Ambiguous range contract",
                subtitle="Ambiguous range",
                rules_primary="No parseable range.",
                floor_strike=None,
                cap_strike=None,
            )
        )


def test_current_six_market_shape_maps_to_contiguous_buckets() -> None:
    markets = pd.DataFrame(
        [
            _market(
                ticker="KXHIGHNY-26JUN02-T73",
                title="Will the high temp in NYC be <73 on Jun 2, 2026?",
                subtitle="72 or below",
                strike_type="less",
                floor_strike=None,
                cap_strike=73,
            ),
            _market(
                ticker="KXHIGHNY-26JUN02-B73.5",
                title="Will the high temp in NYC be 73-74 on Jun 2, 2026?",
                subtitle="73 to 74",
                floor_strike=73,
                cap_strike=74,
            ),
            _market(
                ticker="KXHIGHNY-26JUN02-B75.5",
                title="Will the high temp in NYC be 75-76 on Jun 2, 2026?",
                subtitle="75 to 76",
                floor_strike=75,
                cap_strike=76,
            ),
            _market(
                ticker="KXHIGHNY-26JUN02-B77.5",
                title="Will the high temp in NYC be 77-78 on Jun 2, 2026?",
                subtitle="77 to 78",
                floor_strike=77,
                cap_strike=78,
            ),
            _market(
                ticker="KXHIGHNY-26JUN02-B79.5",
                title="Will the high temp in NYC be 79-80 on Jun 2, 2026?",
                subtitle="79 to 80",
                floor_strike=79,
                cap_strike=80,
            ),
            _market(
                ticker="KXHIGHNY-26JUN02-T80",
                title="Will the high temp in NYC be >80 on Jun 2, 2026?",
                subtitle="81 or above",
                strike_type="greater",
                floor_strike=80,
                cap_strike=None,
            ),
        ]
    )

    result = map_event_contracts(markets, "KXHIGHNY-26JUN02")

    assert result.validation.valid is True
    assert result.validation.bucket_count == 6
    assert result.buckets[0].lower_temp is None
    assert result.buckets[0].upper_temp == 72.5
    assert result.buckets[-1].lower_temp == 80.5
    assert result.buckets[-1].upper_temp is None


def test_missing_interior_bucket_rejects_event_mapping() -> None:
    markets = pd.DataFrame(
        [
            _market(ticker="KXHIGHNY-26JUN02-T73", strike_type="less", floor_strike=None, cap_strike=73),
            _market(ticker="KXHIGHNY-26JUN02-B75.5", floor_strike=75, cap_strike=76),
            _market(ticker="KXHIGHNY-26JUN02-T76", strike_type="greater", floor_strike=76, cap_strike=None),
        ]
    )

    result = map_event_contracts(markets, "KXHIGHNY-26JUN02")

    assert result.validation.valid is False
    assert "invalid_bucket_schema" in result.validation.no_trade_reason


def test_duplicate_bucket_interval_rejects_event_mapping() -> None:
    markets = pd.DataFrame(
        [
            _market(strike_type="less", floor_strike=None, cap_strike=73),
            _market(ticker="A", floor_strike=73, cap_strike=74),
            _market(ticker="B", floor_strike=73, cap_strike=74),
            _market(strike_type="greater", floor_strike=74, cap_strike=None),
        ]
    )

    result = map_event_contracts(markets, "KXHIGHNY-26JUN02")

    assert result.validation.valid is False
    assert result.validation.no_trade_reason == "duplicate_bucket_interval"


def test_mixed_event_tickers_only_map_requested_event() -> None:
    markets = pd.DataFrame(
        [
            _market(event_ticker="KXHIGHNY-26JUN02"),
            _market(event_ticker="KXHIGHNY-26JUN03", ticker="KXHIGHNY-26JUN03-B75.5"),
        ]
    )

    result = map_event_contracts(markets, "KXHIGHNY-26JUN03")

    assert set(result.mapping["event_ticker"]) == {"KXHIGHNY-26JUN03"}


def test_validate_empty_bucket_list_rejects_mapping() -> None:
    validation = validate_contract_mapping([])

    assert validation.valid is False
    assert validation.no_trade_reason == "no_parsed_buckets"
