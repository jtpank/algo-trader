from algo_model import statistics
import numpy as np
import math

class TestStatisticsClass:
    def test_dickey_fuller(self):
        points = np.array([50, 120.83,-25.43,31.72,94.79,61.06,87.51,-65.88,30.40,65.49,15.13,8.23,-18.71,-29.79,-97,42.43,258.70,6.83,45.63,-56.41,-101.23,-143.21,-286.78,-197.79,-236.75])
        expected_t_stat = -1.9161322448371003
        t_stat = statistics.dickey_fuller(points)
        assert math.isclose(t_stat, expected_t_stat)
