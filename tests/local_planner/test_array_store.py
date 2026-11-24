"""NPY tables preserve numeric meaning without unpickling per-row objects."""
import json
import numpy as np
import pytest
import torch

from kappanav.local_planner.data.array_store import ArrayTable, write_table


def test_array_columns_roundtrip_ragged_optional_and_private_writes(tmp_path):
    rows = [dict(name='目标', optional=None, nested=dict(a=True), sequence=(torch.ones(0, 2), [1, 2]),
                 view=np.arange(12, dtype=np.float64).reshape(3, 4).T,
                 scalar=torch.tensor(3.), varying=np.ones((0, 3), np.float32)),
            dict(name='探索', nested=dict(a=False, b=7), sequence=(torch.ones(2, 2), [3]),
                 view=np.arange(12, dtype=np.float64).reshape(3, 4).T,
                 scalar=torch.tensor(4.), varying=np.ones((2, 3), np.float32))]
    metadata = write_table(tmp_path/'table', rows)
    metadata = json.loads(json.dumps(metadata))
    reader = ArrayTable(tmp_path/'table', metadata)
    a, b = reader[0], reader[-1]
    assert a['name'] == '目标' and b['name'] == '探索'
    assert a['optional'] is None and 'optional' not in b
    assert a['nested'] == dict(a=True) and b['nested'] == dict(a=False, b=7)
    assert a['sequence'][1] == [1, 2] and b['sequence'][1] == [3]
    assert isinstance(a['sequence'], tuple) and a['sequence'][0].shape == (0, 2)
    for index, restored in enumerate((a, b)):
        np.testing.assert_array_equal(restored['view'], rows[index]['view'])
        np.testing.assert_array_equal(restored['varying'], rows[index]['varying'])
        torch.testing.assert_close(restored['scalar'], rows[index]['scalar'], rtol=0, atol=0)
        torch.testing.assert_close(restored['sequence'][0], rows[index]['sequence'][0], rtol=0, atol=0)
    assert all(path.suffix == '.npy' for path in (tmp_path/'table').iterdir())
    a['view'][0, 0] = -10
    assert reader[0]['view'][0, 0] == -10
    assert ArrayTable(tmp_path/'table', metadata)[0]['view'][0, 0] == 0
    with pytest.raises(IndexError):
        reader[2]

