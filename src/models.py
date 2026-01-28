from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
    }
