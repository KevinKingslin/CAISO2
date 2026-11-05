"""Narrow compatibility for the unchanged upstream pandas-1 test artifact."""


def enable_legacy_pandas_pickle():
    """Map removed pandas index classes when joblib reads the original artifact.

    All other pickle globals use joblib's original resolver. No data conversion,
    rewritten pickle, or fake import module is needed in the benchmark loader.
    """
    import pandas as pd
    from joblib.numpy_pickle import NumpyUnpickler

    if int(pd.__version__.split(".")[0]) < 2:
        return
    original = NumpyUnpickler.find_class
    if getattr(original, "_caiso_legacy_index_compat", False):
        return

    def find_class(self, module, name):
        if module == "pandas.core.indexes.numeric" and name in {
            "Int64Index",
            "UInt64Index",
            "Float64Index",
        }:
            return pd.Index
        return original(self, module, name)

    find_class._caiso_legacy_index_compat = True
    NumpyUnpickler.find_class = find_class
