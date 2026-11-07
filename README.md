# Оценка качества вина

Приложение на Streamlit использует обученную логистическую регрессию.
Позволяет оценивать одно вино, загружать CSV, менять порог отбора
и просматривать метрики и коэффициенты модели.

Streamlit link: https://wine-quality-app-r94mxzosylbbxbxuapt8tq.streamlit.app/

## Запуск

Версия Python: 3.14.2

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Кэширование

Для модели используется st.cache_resource, поскольку нужен один
общий объект без создания копии при каждом обращении.

Для таблицы и тестовых вероятностей используется st.cache_data.
Результаты вычислений сохраняются, а при обращении возвращается копия.

Без декоратора joblib.load загружал бы модель заново при каждом
действии пользователя.
