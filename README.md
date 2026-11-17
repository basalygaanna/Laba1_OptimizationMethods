# Лабораторная работа № 1 — вариант 18

## Задача
План выпуска мебели: 6 изделий, 4 ограниченных ресурса. Параметры варианта:
`N=1`, `Q=2`, `Y=26`, `V=18`, `seed=10201`.

## Требования
- Python 3.10+;
- HiGHS через `scipy.optimize.linprog(method="highs")`;
- NumPy для независимой проверки;
- pandas и matplotlib для таблиц и графиков.

## Установка и запуск
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
python src/main.py
```

Результаты сохраняются в `outputs/`, графики — в `figures/`.

## Состав
- `data/variant18.json` — данные варианта и описание генерации;
- `src/generate_data.py` — воспроизводимая генерация данных по seed;
- `src/main.py` — загрузка данных, построение и решение модели, независимая проверка, сценарии, отчётность;
- `outputs/base_solution.csv` — найденный план выпуска;
- `outputs/resource_analysis.csv` — расход ресурсов и теневые цены;
- `outputs/scenario_analysis.csv` — сравнение сценариев;
- `outputs/independent_check.json` — проверка без обращения к объекту солвера;
- `outputs/solver.log`, `outputs/solver_log.json` — лог и статусы;
- `outputs/shadow_price_comparison.json` — проверка прогноза теневой цены;
- `figures/*.png` — графики.

## Воспроизводимость
Генератор данных использует `numpy.random.default_rng(10201)`. Модель и данные хранятся раздельно. Для повторного запуска используется одна команда: `python src/main.py`.

