from typing import Callable, List, Optional, Tuple

from data_source import DataSource


class ChangeDetector:
    """
    Generic change detector using repeated sequential tests.
    Works with any test object implementing:
      - reset()
      - update(z_t) -> (stopped: bool, tau: Optional[int])
      - attribute t (time step count)
    """
    def __init__(
        self,
        test_ctor: Callable[[], object],
        N_max: int,
        W: Optional[int] = None
    ):
        """
        Args:
            test_ctor: factory returning a new sequential-test instance
            N_max:     maximum updates before pruning each test
            W:         optional window length; keep only last W tests
        """
        self.test_ctor = test_ctor
        self.N_max = N_max
        self.W = W
        self.active: List[Tuple[object, int]] = []  # (test, start_time)
        self.t = 0

    def reset(self) -> None:
        """Clear all active tests and reset time counter."""
        self.active.clear()
        self.t = 0

    def update(self, z_t) -> Optional[int]:
        """
        Process one new observation.

        Returns:
            detection_time if any test stops, otherwise None.
        """
        self.t += 1

        # Spawn a new test at current time
        new_test = self.test_ctor()
        new_test.reset()
        stopped, _ = new_test.update(z_t)
        if stopped:
            return self.t

        # Add to active list and enforce window
        self.active.append((new_test, self.t))
        if self.W is not None:
            while len(self.active) > self.W:
                self.active.pop(0)

        # Update all active tests and prune
        survivors: List[Tuple[object, int]] = []
        for test, start_time in self.active:
            stopped, _ = test.update(z_t)
            if stopped:
                return self.t
            if test.t < self.N_max:
                survivors.append((test, start_time))
        self.active = survivors
        return None

    def run_on_source(self, src: DataSource) -> Tuple[bool, int]:
        """
        Run the detector on a streaming DataSource.

        Returns:
            (alarm_fired: bool, stop_time: int)
        """
        src.reset()
        self.reset()
        for z in src:
            alarm_t = self.update(z)
            if alarm_t is not None:
                return True, alarm_t
        return False, self.t


# ----------------------------------------------------------------------------
# To copy into data_source.py:
# ----------------------------------------------------------------------------

class ChangePointSource(DataSource):
    """
    DataSource that switches from a pre-change source to a post-change source at T_c.
    """
    def __init__(
        self,
        pre_source: DataSource,
        post_source: DataSource,
        T_c: int,
        length: Optional[int] = None,
        randomize_Tc: bool = False
    ):
        self.pre_source = pre_source
        self.post_source = post_source
        self.T_c = T_c
        self.length = length
        self.randomize_Tc = randomize_Tc
        self._t = 0

    def reset(self) -> None:
        """Reset both sources and optionally randomize T_c."""
        self.pre_source.reset()
        self.post_source.reset()
        self._t = 0
        if self.randomize_Tc and self.length is not None:
            import numpy as np
            self.T_c = np.random.randint(1, self.length + 1)

    def __iter__(self):
        self.reset()
        while True:
            self._t += 1
            if self.length is not None and self._t > self.length:
                break
            if self._t <= self.T_c:
                yield next(self.pre_source)
            else:
                yield next(self.post_source)
