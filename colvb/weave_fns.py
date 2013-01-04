# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
from scipy import weave

def LDA_mult(X,y,target=None):
    """Perform matrix multiplication  X*y, subject to X being a sparse vector containing only zeros and ones."""
    if target is None:
        target = np.zeros((X.shape[0],y.shape[1]))
    rows = X.row
    cols = X.col
    code = """
    PyObject *itr;
    int *row;
    int *col;
    itr = PyArray_MultiIterNew(2, rows_array, cols_array);
    while(PyArray_MultiIter_NOTDONE(itr)) {
        row = (int *) PyArray_MultiIter_DATA(itr, 0);
        col = (int *) PyArray_MultiIter_DATA(itr, 1);
        for(int i=0;i<y_array->dimensions[1];i++){
            target(*row,i) += y(*col,i);
        }
        PyArray_MultiIter_NEXT(itr);
    }
    Py_DECREF(itr);
    """
    weave.inline(code,['rows','cols','y','target'],type_converters=weave.converters.blitz)
    return target
