from time import time

def test_run_tree(model, X, y):

    st = time()
    model.fit(X, y)
    en = time()

    stp = time()
    y_pred = model.predict(X)
    enp = time()

    return (y_pred,en-st,enp-stp)
