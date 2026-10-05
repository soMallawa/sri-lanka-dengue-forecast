from dengue_forecast.config import load_district_aliases
from dengue_forecast.sources.config import REQUIRED_SOURCE_KEYS, _load_sources


def test_source_configuration_is_available():
    assert set(_load_sources()) == set(REQUIRED_SOURCE_KEYS)


def test_district_alias_configuration_is_available():
    assert len({value[0] for value in load_district_aliases().values()}) == 25
