import numpy as np
from dataclasses import dataclass, field
from typing import Union
from utils.utils import format_csv_time
import pandas as pd

Number = Union[int, float, np.number]

## dataclass definitions

@dataclass
class Displacement:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    time: np.ndarray
    n: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, mapping: dict):
        return cls(
            x=df[mapping["x"]].to_numpy(),
            y=df[mapping["y"]].to_numpy(),
            z=df[mapping["z"]].to_numpy(),
            time=df[mapping["time"]].to_numpy(),
        )

    @classmethod
    def from_sd_card(cls, path: str):
        mapping = {
            "x": " x (m)",
            "y": " y(m)",
            "z": " z(m)",
            "time": "time",
        }

        df = pd.read_csv(path)
        df["time"] = format_csv_time(df)

        return cls.from_dataframe(df, mapping)

@dataclass
class DirectionalMoments:
    """
    Container for directional moment arrays (typically a1, b1, a2, b2).

    Notes:
    - These are usually *derived* (normalized) quantities, so they generally should
      NOT be added or scaled directly. Recompute them from averaged spectra instead.
    """
    a1: np.ndarray = field(default_factory=lambda: np.array([]))
    b1: np.ndarray = field(default_factory=lambda: np.array([]))
    a2: np.ndarray = field(default_factory=lambda: np.array([]))
    b2: np.ndarray = field(default_factory=lambda: np.array([]))

    def append(self, other: "DirectionalMoments") -> None:
        """
        Append another DirectionalMoments object along the leading (time/segment) axis.
        Allows mixing 1D (F,) and 2D (T,F) arrays by promoting 1D to (1,F).
        """
        def normalize_array(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            if x.size == 0:
                return x
            if x.ndim == 0:
                return np.array([x])
            if x.ndim == 1:
                return x.reshape(1, -1)
            return x

        def concat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a = np.asarray(a)
            b = np.asarray(b)

            if a.size == 0:
                return normalize_array(b)
            if b.size == 0:
                return normalize_array(a)

            a_norm = normalize_array(a)
            b_norm = normalize_array(b)

            if a_norm.shape[1:] != b_norm.shape[1:]:
                raise ValueError(f"Incompatible shapes: {a.shape} vs {b.shape}")

            return np.concatenate((a_norm, b_norm), axis=0)

        self.a1 = concat(self.a1, other.a1)
        self.b1 = concat(self.b1, other.b1)
        self.a2 = concat(self.a2, other.a2)
        self.b2 = concat(self.b2, other.b2)

    def reshape_for_solver(self) -> "DirectionalMoments":
        """
        Reshape all moment arrays to 2D shape [n_points, n_frequencies].
        - (F,)     -> (1,F)
        - (T,F)    -> (T,F)
        - (...,F)  -> (-1,F)  (flatten leading dims)
        """
        def shape_moment(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr)
            if arr.size == 0:
                return arr
            if arr.ndim == 1:
                return arr[None, :]
            if arr.ndim == 2:
                return arr
            return arr.reshape(-1, arr.shape[-1])

        return DirectionalMoments(
            a1=shape_moment(self.a1),
            b1=shape_moment(self.b1),
            a2=shape_moment(self.a2),
            b2=shape_moment(self.b2),
        )



@dataclass
class Spectrum:
    frequency: np.ndarray
    ezz: np.ndarray
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    direction: np.ndarray = field(default_factory=lambda: np.array([]))
    exx: np.ndarray = field(default_factory=lambda: np.array([]))
    eyy: np.ndarray = field(default_factory=lambda: np.array([]))
    enn: np.ndarray = field(default_factory=lambda: np.array([]))
    cxy: np.ndarray = field(default_factory=lambda: np.array([]))
    czn: np.ndarray = field(default_factory=lambda: np.array([]))
    qxz: np.ndarray = field(default_factory=lambda: np.array([]))
    qyz: np.ndarray = field(default_factory=lambda: np.array([]))
    directional_moments: DirectionalMoments = field(default_factory=DirectionalMoments)
    directional_spectra: np.ndarray = field(default_factory=lambda: np.array([]))

    def append(self, other: 'Spectrum') -> None:
        """Appends another Spectrum object to this one along the time dimension."""
        if not np.array_equal(self.frequency, other.frequency):
            raise ValueError("Frequencies must match to append spectra.")

        def normalize_array(x: np.ndarray, target_ndim: int) -> np.ndarray:
            """Ensure array is at least `target_ndim` by adding leading dims."""
            if np.isscalar(x) or x.ndim == 0:
                x = np.array([x])
            while x.ndim < target_ndim:
                x = np.expand_dims(x, axis=0)
            return x

        def concat(a: np.ndarray, b: np.ndarray, target_ndim: int) -> np.ndarray:
            """Concatenate a and b along time dimension (axis=0)."""
            if a.size == 0:
                return normalize_array(b, target_ndim)
            if b.size == 0:
                return normalize_array(a, target_ndim)

            a_norm = normalize_array(a, target_ndim)
            b_norm = normalize_array(b, target_ndim)

            if a_norm.shape[1:] != b_norm.shape[1:]:
                raise ValueError(f"Incompatible shapes for concatenation: {a.shape} vs {b.shape}")
            return np.concatenate((a_norm, b_norm), axis=0)

        # Spectral variables (1D or 2D → stack along time axis)
        self.ezz = concat(self.ezz, other.ezz, target_ndim=2)
        self.exx = concat(self.exx, other.exx, target_ndim=2)
        self.eyy = concat(self.eyy, other.eyy, target_ndim=2)
        self.enn = concat(self.enn, other.enn, target_ndim=2)
        self.cxy = concat(self.cxy, other.cxy, target_ndim=2)
        self.czn = concat(self.czn, other.czn, target_ndim=2)
        self.qxz = concat(self.qxz, other.qxz, target_ndim=2)
        self.qyz = concat(self.qyz, other.qyz, target_ndim=2)

        # Time (scalar or 1D → stacked as (T,))
        self.time = concat(self.time, other.time, target_ndim=1)

        # Directional spectra (2D or 3D → stacked along time axis)
        self.directional_spectra = concat(self.directional_spectra, other.directional_spectra, target_ndim=3)

        # Directional moments: assume it's a list of dicts or similar
        self.directional_moments.append(other.directional_moments)

    def _require_same_frequency(self, other: "Spectrum") -> None:
        if not np.array_equal(self.frequency, other.frequency):
            raise ValueError("Frequencies must match to combine spectra.")

    @staticmethod
    def _as_array(x: np.ndarray) -> np.ndarray:
        # normalize None-like / scalar-ish into ndarray
        if x is None:
            return np.array([])
        return np.asarray(x)

    @staticmethod
    def _add_optional(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Add arrays if both non-empty; otherwise return the non-empty one (copy).
        Supports:
          - (F,) + (F,)
          - (T,F) + (T,F)
          - (T,F) + (F,)  (broadcast across time)
          - (F,) + (T,F)  (broadcast across time)
        """
        a = Spectrum._as_array(a)
        b = Spectrum._as_array(b)

        if a.size == 0 and b.size == 0:
            return np.array([])
        if a.size == 0:
            return b.copy()
        if b.size == 0:
            return a.copy()

        # If either is 1D and the other is 2D (time-stacked), broadcast 1D across time.
        if a.ndim == 1 and b.ndim == 2:
            if a.shape[0] != b.shape[1]:
                raise ValueError(f"Incompatible shapes for add: {a.shape} vs {b.shape}")
            a = np.broadcast_to(a, b.shape)
        elif a.ndim == 2 and b.ndim == 1:
            if b.shape[0] != a.shape[1]:
                raise ValueError(f"Incompatible shapes for add: {a.shape} vs {b.shape}")
            b = np.broadcast_to(b, a.shape)

        if a.shape != b.shape:
            raise ValueError(f"Incompatible shapes for add: {a.shape} vs {b.shape}")
        return a + b

    @staticmethod
    def _scale_optional(a: np.ndarray, k: Number) -> np.ndarray:
        a = Spectrum._as_array(a)
        if a.size == 0:
            return np.array([])
        return a * k

    def __add__(self, other: "Spectrum") -> "Spectrum":
        if not isinstance(other, Spectrum):
            return NotImplemented
        self._require_same_frequency(other)

        ezz_sum = self._add_optional(self.ezz, other.ezz)

        out = Spectrum(
            frequency=self.frequency.copy(),
            ezz=ezz_sum,
            time=self.time.copy() if self.time.size else np.array([]),
            direction=self.direction.copy() if self.direction.size else np.array([]),
            exx=self._add_optional(self.exx, other.exx),
            eyy=self._add_optional(self.eyy, other.eyy),
            enn=self._add_optional(self.enn, other.enn),
            cxy=self._add_optional(self.cxy, other.cxy),
            czn=self._add_optional(self.czn, other.czn),
            qxz=self._add_optional(self.qxz, other.qxz),
            qyz=self._add_optional(self.qyz, other.qyz),
            directional_moments=self.directional_moments,
            directional_spectra=self.directional_spectra,
        )
        return out

    def __iadd__(self, other: "Spectrum") -> "Spectrum":
        if not isinstance(other, Spectrum):
            return NotImplemented
        self._require_same_frequency(other)

        self.ezz = self._add_optional(self.ezz, other.ezz)
        self.exx = self._add_optional(self.exx, other.exx)
        self.eyy = self._add_optional(self.eyy, other.eyy)
        self.enn = self._add_optional(self.enn, other.enn)
        self.cxy = self._add_optional(self.cxy, other.cxy)
        self.czn = self._add_optional(self.czn, other.czn)
        self.qxz = self._add_optional(self.qxz, other.qxz)
        self.qyz = self._add_optional(self.qyz, other.qyz)
        self.directional_spectra = self._add_optional(self.directional_spectra, other.directional_spectra)
        return self

    def __mul__(self, k: Number) -> "Spectrum":
        if not isinstance(k, (int, float, np.number)):
            return NotImplemented

        return Spectrum(
            frequency=self.frequency.copy(),
            ezz=self._scale_optional(self.ezz, k),
            time=self.time.copy() if self.time.size else np.array([]),
            direction=self.direction.copy() if self.direction.size else np.array([]),
            exx=self._scale_optional(self.exx, k),
            eyy=self._scale_optional(self.eyy, k),
            enn=self._scale_optional(self.enn, k),
            cxy=self._scale_optional(self.cxy, k),
            czn=self._scale_optional(self.czn, k),
            qxz=self._scale_optional(self.qxz, k),
            qyz=self._scale_optional(self.qyz, k),
            directional_moments=self.directional_moments,
            directional_spectra=self._scale_optional(self.directional_spectra, k),
        )


    def __rmul__(self, k: Number) -> "Spectrum":
        return self.__mul__(k)

    def __imul__(self, k: Number) -> "Spectrum":
        if not isinstance(k, (int, float, np.number)):
            return NotImplemented

        self.ezz = self._scale_optional(self.ezz, k)
        self.exx = self._scale_optional(self.exx, k)
        self.eyy = self._scale_optional(self.eyy, k)
        self.enn = self._scale_optional(self.enn, k)
        self.cxy = self._scale_optional(self.cxy, k)
        self.czn = self._scale_optional(self.czn, k)
        self.qxz = self._scale_optional(self.qxz, k)
        self.qyz = self._scale_optional(self.qyz, k)
        self.directional_spectra = self._scale_optional(self.directional_spectra, k)
        return self