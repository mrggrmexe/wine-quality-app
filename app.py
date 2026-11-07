import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Оценка вина", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model.joblib")


@st.cache_data
def load_test():
    return pd.read_csv(BASE_DIR / "test.csv")


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test = load_test()
X_test = test.drop(columns="target")
y_test = test["target"]
features = X_test.columns.tolist()
proba = test_proba(model, X_test)

config = json.loads(
    (BASE_DIR / "config.json").read_text(encoding="utf-8")
)

st.title("Оценка качества вина")

threshold = st.sidebar.slider(
    "Порог отбора",
    min_value=0.0,
    max_value=1.0,
    value=float(config["threshold"]),
    step=0.001,
    format="%.3f",
    key="threshold"
)

st.sidebar.write(
    "Вино попадает в подборку, если его вероятность "
    "не ниже выбранного порога."
)

tab_one, tab_metrics, tab_batch, tab_weights = st.tabs([
    "Одно вино",
    "Модельные метрики",
    "Пакетное предсказание",
    "Что выучила модель"
])


with tab_one:
    st.subheader("Характеристики вина")

    values = {}
    columns = st.columns(3)

    for i, feature in enumerate(features):
        with columns[i % 3]:
            if feature == "is_red":
                values[feature] = int(st.toggle("Красное вино"))
            else:
                values[feature] = st.number_input(
                    feature,
                    min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()),
                    value=float(X_test[feature].median()),
                    format="%.5f"
                )

    if st.button("Оценить вино"):
        row = pd.DataFrame([values], columns=features)
        probability = model.predict_proba(row)[0, 1]

        st.metric("Вероятность хорошего вина", f"{probability:.1%}")

        if probability >= threshold:
            st.success("Вино попадает в подборку")
        else:
            st.info("Вино не попадает в подборку")


with tab_metrics:
    st.subheader("Качество на тестовой выборке")

    pred = (proba >= threshold).astype(int)

    precision = metrics.precision_score(
        y_test, pred, zero_division=0
    )
    recall = metrics.recall_score(
        y_test, pred, zero_division=0
    )
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    columns = st.columns(3)
    columns[0].metric("Precision", f"{precision:.3f}")
    columns[1].metric("Recall", f"{recall:.3f}")
    columns[2].metric("F1", f"{f1:.3f}")

    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])

    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Остальные", "Хорошие"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Настоящий класс")
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    p_curve, r_curve, _ = metrics.precision_recall_curve(
        y_test, proba
    )

    tn, fp, fn, tp = cm.ravel()
    current_fpr = fp / (fp + tn) if fp + tn > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].scatter(current_fpr, recall, color="red", zorder=3)
    axes[0].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC-кривая"
    )

    axes[1].plot(r_curve, p_curve)
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray")
    axes[1].scatter(recall, precision, color="red", zorder=3)
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="PR-кривая"
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        "При повышении порога подборка становится меньше, "
        "обычно в неё попадает меньше посредственных вин. "
        "При снижении порога модель находит больше хороших вин, "
        "но чаще добавляет посредственные."
    )


with tab_batch:
    st.subheader("Загрузка CSV с характеристиками вин")
    st.write("Нужные столбцы: " + ", ".join(features))

    uploaded = st.file_uploader("Выберите CSV", type=["csv"])

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            missing = sorted(set(features) - set(data.columns))

            if missing:
                st.error("Не хватает столбцов: " + ", ".join(missing))
            elif data.empty:
                st.error("В файле нет строк с данными")
            else:
                X_batch = data[features].apply(
                    pd.to_numeric, errors="raise"
                )

                if not np.isfinite(X_batch.to_numpy()).all():
                    st.error(
                        "Признаки содержат пропуски или бесконечные значения"
                    )
                elif not X_batch["is_red"].isin([0, 1]).all():
                    st.error("Столбец is_red должен содержать только 0 и 1")
                else:
                    batch_proba = model.predict_proba(X_batch)[:, 1]

                    result = data.copy()
                    result["probability"] = batch_proba
                    result["prediction"] = (
                        batch_proba >= threshold
                    ).astype(int)

                    st.dataframe(result)

                    st.download_button(
                        "Скачать результат",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name="wine_predictions.csv",
                        mime="text/csv"
                    )

        except (ValueError, pd.errors.ParserError, UnicodeError) as error:
            st.error(f"Не удалось обработать CSV: {error}")


with tab_weights:
    st.subheader("Коэффициенты модели")

    coefficients = pd.Series(
        model.steps[-1][1].coef_[0],
        index=features
    ).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        coefficients.index,
        coefficients.values,
        color=[
            "seagreen" if value > 0 else "salmon"
            for value in coefficients.values
        ]
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Коэффициент после масштабирования")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)

    st.write("Сильнее всего за хорошее вино: " + ", ".join(positive.index))
    st.write("Сильнее всего против: " + ", ".join(negative.index))
    st.caption(
        "Коэффициенты показывают связи, которые нашла модель, "
        "а не причинное влияние признаков."
    )