from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from uncert_ident.methods.classification import get_test_data, FEATURE_KEYS
from uncert_ident.methods.classification import confusion_matrix


feat, labl = get_test_data("PH-Xiao-15", FEATURE_KEYS, 1)
# feat, labl = get_test_data("NACA4412-Vinuesa-top-1", FEATURE_KEYS, 1)
print(labl.shape)

strategies = ["most_frequent", "stratified", "constant", "uniform", "stratified"]
dummies = [DummyClassifier(strategy=strat, constant=1) for strat in strategies]


for dum, strat in zip(dummies, strategies):
    print(f"\n\nDummy with strategy {strat}")
    dum.fit(feat, labl)
    print("F1-Score:\t", f1_score(labl, dum.predict(feat)))
    print("Confusion matrix:", confusion_matrix(dum.predict(feat), labl, return_list=False))
