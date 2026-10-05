import pandas as pd
import pytest
from dengue_forecast.public_training import _training_rows_available_by_cutoff, PublicTrainingError


def test_training_cutoff_excludes_unfinished_target_week():
    # Synthetic origin: Jan 6, H1 target Jan 13-19; Jan 13 is not its completed label.
    frame = pd.DataFrame({'week_start_date': ['2024-01-06']})
    with pytest.raises(PublicTrainingError, match='No rows'):
        _training_rows_available_by_cutoff(frame, horizon=1,
            label_cutoff=pd.Timestamp('2024-01-13'), allow_2025_research_scope=False)
    kept = _training_rows_available_by_cutoff(frame, horizon=1,
        label_cutoff=pd.Timestamp('2024-01-19'), allow_2025_research_scope=False)
    assert len(kept) == 1
