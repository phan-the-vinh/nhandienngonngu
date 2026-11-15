from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =====================================================
# 1. DUONG DAN
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2] / "ai"

DATASET_PATH = (
    BASE_DIR
    / "dataset"
    / "dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "mlp_static_mirror.pkl"
)


# =====================================================
# 2. DOC DATASET
# =====================================================

if not DATASET_PATH.exists():
    print("Khong tim thay dataset:")
    print(DATASET_PATH)
    exit()


df = pd.read_csv(DATASET_PATH)


print("\n===== DATASET GOC =====")
print("Tong so mau:", len(df))

print("\nSo mau tung ky tu:")
print(
    df["label"]
    .value_counts()
    .sort_index()
)


# =====================================================
# 3. LAY 63 DAC TRUNG
# =====================================================

feature_columns = [
    f"f{i}"
    for i in range(1, 64)
]


X = df[feature_columns].copy()

y = df["label"].copy()


# =====================================================
# 4. CHIA TRAIN / TEST TRUOC
# =====================================================
#
# Chia du lieu goc truoc.
# Chi mirror tap TRAIN.
# Tap TEST van giu nguyen.
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n===== CHIA DU LIEU =====")
print("Train goc:", len(X_train))
print("Test goc :", len(X_test))


# =====================================================
# 5. TAO DU LIEU MIRROR
# =====================================================
#
# Moi landmark:
#
# x, y, z
#
# f1  = x0
# f2  = y0
# f3  = z0
#
# f4  = x1
# f5  = y1
# f6  = z1
#
# ...
#
# Sau normalize, co tay da la goc.
# Ta phan chieu theo truc X:
#
# x -> -x
# y -> giu nguyen
# z -> giu nguyen
# =====================================================

X_train_mirror = X_train.copy()


x_columns = [
    f"f{i}"
    for i in range(1, 64, 3)
]


for column in x_columns:

    X_train_mirror[column] = (
        -X_train_mirror[column]
    )


# Nhan cua ban mirror van la cung ky tu
y_train_mirror = y_train.copy()


# =====================================================
# 6. GHEP DU LIEU GOC + MIRROR
# =====================================================

X_train_augmented = pd.concat(
    [
        X_train,
        X_train_mirror
    ],
    ignore_index=True
)


y_train_augmented = pd.concat(
    [
        y_train,
        y_train_mirror
    ],
    ignore_index=True
)


print("\n===== MIRROR AUGMENTATION =====")

print(
    "Train goc   :",
    len(X_train)
)

print(
    "Train mirror:",
    len(X_train_mirror)
)

print(
    "Tong train  :",
    len(X_train_augmented)
)


print("\nSo mau sau khi mirror:")

print(
    y_train_augmented
    .value_counts()
    .sort_index()
)


# =====================================================
# 7. TAO MLP
# =====================================================

model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "mlp",
        MLPClassifier(

            hidden_layer_sizes=(64, 32),

            activation="relu",

            solver="adam",

            max_iter=2000,

            random_state=42
        )
    )
])


# =====================================================
# 8. TRAIN
# =====================================================

print("\nDang train MLP MIRROR...")


model.fit(
    X_train_augmented,
    y_train_augmented
)


print("Train xong.")


# =====================================================
# 9. TEST
# =====================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n===== KET QUA MIRROR =====")

print(
    f"Accuracy: {accuracy:.4f}"
)


print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =====================================================
# 10. LUU MODEL
# =====================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


print("\nDa luu model MIRROR:")
print(MODEL_PATH)
