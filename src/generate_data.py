import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 10201

def generate():
    rng = np.random.default_rng(SEED)
    products = ["Стул", "Стол", "Шкаф", "Полка", "Кровать", "Письменный стол"]
    resources = ["Древесина, м³", "Трудозатраты, ч", "Лакокрасочные материалы, л", "Станочное время, ч"]
    coefficients = np.array([
        rng.uniform(0.025, 0.09, 6),
        rng.uniform(1.5, 5.5, 6),
        rng.uniform(0.15, 0.65, 6),
        rng.uniform(0.4, 1.8, 6)
    ]).round(3)
    profits = rng.integers(1800, 6501, 6).astype(float).tolist()
    return {
        "variant": 18, "N": 1, "Q": 2, "Y": 26, "seed": SEED,
        "generation": "numpy.random.default_rng(seed=10201); resource coefficients sampled uniformly from declared ranges and rounded to 0.001; unit profits sampled as integers.",
        "products": products, "resources": resources,
        "resource_coefficients": coefficients.tolist(),
        "unit_profit": profits,
        "capacities": [2.6, 150.0, 20.0, 55.0],
        "minimum_total_order": 30.0,
        "units": {
            "resource_coefficients": ["м³/шт.", "ч/шт.", "л/шт.", "ч/шт."],
            "capacities": ["м³", "ч", "л", "ч"],
            "unit_profit": "денежных единиц/шт."
        },
        "assumptions": [
            "Все изделия однородны внутри типа; выпуск задаётся непрерывными переменными (усреднённый план).",
            "Прибыль на единицу постоянна и линейна по объёму выпуска.",
            "Расход каждого ресурса на единицу изделия постоянен.",
            "Не учитываются переналадки, складские ограничения, брак и дискретность изделий.",
            "Минимальный подтверждённый суммарный заказ составляет 30 изделий; это дополнительное учебное допущение."
        ]
    }

if __name__ == "__main__":
    target = ROOT / "data" / "variant18.json"
    target.write_text(json.dumps(generate(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Данные варианта записаны: {target}")
