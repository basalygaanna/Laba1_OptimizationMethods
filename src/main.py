from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "variant18.json"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
TOL = 1e-7


def load_data():
    """Read the instance data; model logic stays separate from JSON."""
    return json.loads(DATA.read_text(encoding="utf-8"))


def build_model(data, capacities=None, profits=None):
    """Build max profit with four resource constraints and x >= 0."""
    A = np.asarray(data["resource_coefficients"], dtype=float)
    b = np.asarray(capacities if capacities is not None else data["capacities"], dtype=float)
    c = np.asarray(profits if profits is not None else data["unit_profit"], dtype=float)
    # scipy.linprog minimizes, so maximizing c @ x is minimizing -c @ x.
    result = linprog(
        c=-c,
        A_ub=A,
        b_ub=b,
        bounds=[(0, None)] * len(c),
        method="highs-ds",
        options={"presolve": True}
    )
    return result, A, b, c


def independent_check(x, A, b, profits):
    """Recalculate feasibility and objective using NumPy only."""
    resource_use = A @ x
    resource_slack = b - resource_use
    violations = []
    if np.any(resource_slack < -TOL):
        violations.append("Превышен лимит одного или нескольких ресурсов")
    if np.any(x < -TOL):
        violations.append("Есть отрицательные объёмы выпуска")
    return {
        "feasible": not violations,
        "violations": violations,
        "resource_use": resource_use.tolist(),
        "resource_slack": resource_slack.tolist(),
        "total_output": float(np.sum(x)),
        "objective_recalculated": float(np.asarray(profits) @ x),
        "max_resource_violation": float(max(0.0, -np.min(resource_slack))),
        "tolerance": TOL
    }


def solve_scenario(data, name, capacities):
    result, A, b, profits = build_model(data, capacities)
    record = {
        "scenario": name,
        "status": int(result.status),
        "solver_message": result.message,
        "success": bool(result.success),
        "capacities": b.tolist()
    }
    if result.success:
        x = np.asarray(result.x, dtype=float)
        check = independent_check(x, A, b, profits)
        record.update({
            "objective_solver": float(-result.fun),
            "production": x.tolist(),
            "independent_check": check,
            "resource_shadow_prices": (-np.asarray(result.ineqlin.marginals)).tolist(),
            "resource_slacks": np.asarray(result.ineqlin.residual).tolist(),
            "active_resources": [
                data["resources"][i]
                for i, slack in enumerate(result.ineqlin.residual)
                if abs(slack) <= TOL
            ]
        })
    return result, record


def sensitivity_ranges(data, base_result, A, b, profits):
    """
    One-at-a-time objective-coefficient ranges for which the current optimal
    basis remains dual-feasible. This is the classical RHS-fixed objective
    coefficient sensitivity interval.
    """
    x = np.asarray(base_result.x, dtype=float)
    slacks = np.asarray(base_result.ineqlin.residual, dtype=float)
    basic_cols = np.flatnonzero(x > TOL)
    active_rows = np.flatnonzero(np.abs(slacks) <= TOL)

    if len(basic_cols) != len(active_rows):
        raise RuntimeError("Для данной версии анализа число базисных переменных не совпало с числом активных ресурсов.")
    B = A[np.ix_(active_rows, basic_cols)]
    if np.linalg.matrix_rank(B) != len(basic_cols):
        raise RuntimeError("Матрица текущего базиса вырождена; интервалы нельзя получить этим упрощённым расчётом.")

    # Dual prices for active resource constraints: B.T @ lambda = c_B.
    lam = np.linalg.solve(B.T, profits[basic_cols])
    nonbasic_cols = [j for j in range(len(profits)) if j not in set(basic_cols)]
    rows = []

    for j in range(len(profits)):
        delta_low, delta_high = -np.inf, np.inf
        if j in basic_cols:
            basis_position = int(np.where(basic_cols == j)[0][0])
            unit = np.zeros(len(basic_cols))
            unit[basis_position] = 1.0
            dlam = np.linalg.solve(B.T, unit)

            # Dual prices must remain nonnegative for max with <= constraints.
            for q in range(len(lam)):
                alpha, beta = dlam[q], lam[q]
                if abs(alpha) > 1e-12:
                    bound = -beta / alpha
                    if alpha > 0:
                        delta_low = max(delta_low, bound)
                    else:
                        delta_high = min(delta_high, bound)

            # Reduced costs of all nonbasic variables must remain <= 0.
            for k in nonbasic_cols:
                beta = profits[k] - A[active_rows, k] @ lam
                alpha = -(A[active_rows, k] @ dlam)
                if abs(alpha) > 1e-12:
                    bound = -beta / alpha
                    if alpha > 0:
                        delta_high = min(delta_high, bound)
                    else:
                        delta_low = max(delta_low, bound)
        else:
            # Only c_j changes; its reduced cost must stay nonpositive.
            upper_delta = float(A[active_rows, j] @ lam - profits[j])
            delta_high = min(delta_high, upper_delta)

        lower = profits[j] + delta_low
        upper = profits[j] + delta_high
        rows.append({
            "Изделие": data["products"][j],
            "Текущая прибыль": float(profits[j]),
            "Нижняя граница": None if not np.isfinite(lower) else float(lower),
            "Верхняя граница": None if not np.isfinite(upper) else float(upper),
            "Тип переменной": "базисная" if j in basic_cols else "небазисная",
            "Интерпретация": (
                "При изменении только этой прибыли в указанном интервале текущий базис остаётся оптимальным."
            )
        })

    return rows, {
        "active_resources": [data["resources"][i] for i in active_rows],
        "basic_products": [data["products"][i] for i in basic_cols],
        "dual_prices_active_constraints": lam.tolist(),
        "method": "Параметрический анализ текущего базиса: неотрицательность двойственных цен и неположительность reduced costs.",
        "note": "Границы относятся к изменению одного коэффициента прибыли за раз при неизменных остальных коэффициентах и ресурсных лимитах."
    }


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    data = load_data()
    caps = np.asarray(data["capacities"], dtype=float)

    scenarios = [
        ("Базовый", caps),
        ("Дефицитный: древесина −20%", caps * np.array([0.8, 1.0, 1.0, 1.0])),
        ("Стрессовый: древесина и труд −50%", caps * np.array([0.5, 0.5, 1.0, 1.0]))
    ]

    records, solved = [], {}
    for name, scenario_caps in scenarios:
        result, rec = solve_scenario(data, name, scenario_caps)
        records.append(rec)
        if result.success:
            solved[name] = (result, scenario_caps)

    base_result, base_caps = solved["Базовый"]
    x = np.asarray(base_result.x)
    A = np.asarray(data["resource_coefficients"], dtype=float)
    profits = np.asarray(data["unit_profit"], dtype=float)
    check = independent_check(x, A, base_caps, profits)

    # Base production table.
    products_df = pd.DataFrame({
        "Изделие": data["products"],
        "Выпуск, шт.": x,
        "Прибыль на единицу, ден. ед./шт.": profits,
        "Прибыль, ден. ед.": x * profits
    })
    products_df.to_csv(OUT / "base_solution.csv", index=False, encoding="utf-8-sig")

    # Resource usage, remaining capacities, and shadow prices.
    resources_df = pd.DataFrame({
        "Ресурс": data["resources"],
        "Лимит": base_caps,
        "Расход": A @ x,
        "Остаток": base_caps - A @ x,
        "Теневая цена, ден. ед./ед. ресурса": -np.asarray(base_result.ineqlin.marginals)
    })
    resources_df.to_csv(OUT / "resource_analysis.csv", index=False, encoding="utf-8-sig")

    # Scenario comparison.
    scenario_rows = []
    for rec in records:
        row = {"Сценарий": rec["scenario"], "Статус": rec["solver_message"]}
        if rec["success"]:
            row.update({
                "Прибыль, ден. ед.": rec["objective_solver"],
                "Всего изделий, шт.": sum(rec["production"]),
                "Выпуск по изделиям": "; ".join(
                    f"{p}: {v:.3f}" for p, v in zip(data["products"], rec["production"])
                ),
                "Независимая проверка": "пройдена" if rec["independent_check"]["feasible"] else "ошибка"
            })
        else:
            row.update({
                "Прибыль, ден. ед.": None,
                "Всего изделий, шт.": None,
                "Выпуск по изделиям": "Нет допустимого решения",
                "Независимая проверка": "не применима: infeasible"
            })
        scenario_rows.append(row)
    pd.DataFrame(scenario_rows).to_csv(OUT / "scenario_analysis.csv", index=False, encoding="utf-8-sig")

    # Detailed machine-readable results.
    (OUT / "solver_log.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "independent_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compare the shadow-price prediction with the actual 20% wood reduction.
    deficit_result = solved["Дефицитный: древесина −20%"][0]
    delta_wood = -0.2 * caps[0]
    wood_shadow = -float(base_result.ineqlin.marginals[0])
    predicted_delta = wood_shadow * delta_wood
    actual_delta = float(-deficit_result.fun - (-base_result.fun))
    shadow_comparison = {
        "wood_shadow_price": wood_shadow,
        "wood_capacity_change": delta_wood,
        "predicted_profit_change": predicted_delta,
        "actual_profit_change": actual_delta,
        "absolute_difference": abs(predicted_delta - actual_delta)
    }
    (OUT / "shadow_price_comparison.json").write_text(
        json.dumps(shadow_comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Optional assignment 4: sensitivity ranges of objective coefficients.
    ranges, sensitivity_meta = sensitivity_ranges(data, base_result, A, base_caps, profits)
    pd.DataFrame(ranges).to_csv(OUT / "sensitivity_ranges.csv", index=False, encoding="utf-8-sig")
    (OUT / "sensitivity_ranges.json").write_text(
        json.dumps({"metadata": sensitivity_meta, "ranges": ranges}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 1. Production plan: horizontal bars make product names readable.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y_pos = np.arange(len(data["products"]))
    bars = ax.barh(y_pos, x)
    ax.set_yticks(y_pos, labels=data["products"])
    ax.invert_yaxis()
    ax.set_xlabel("Выпуск, шт.")
    ax.set_title("Оптимальный план выпуска мебели")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, x):
        ax.text(value + 0.35, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, max(x) * 1.16 if max(x) > 0 else 1)
    fig.tight_layout()
    fig.savefig(FIG / "production_plan.png", dpi=180)
    plt.close(fig)

    # 2. Show resource utilization in percent: resources have different units.
    use = A @ x
    utilization = np.divide(use, base_caps, out=np.zeros_like(use), where=base_caps != 0) * 100
    fig, ax = plt.subplots(figsize=(10, 5.8))
    y_pos = np.arange(len(data["resources"]))
    bars = ax.barh(y_pos, utilization)
    ax.set_yticks(y_pos, labels=data["resources"])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Использование ресурса, % от доступного лимита")
    ax.set_title("Степень использования ресурсов в базовом сценарии")
    ax.axvline(100, linestyle="--", linewidth=1, label="Полное использование лимита")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, utilization):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%", va="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "resource_usage.png", dpi=180)
    plt.close(fig)

    # 3. Profit comparison across the three required scenarios.
    feasible_records = [r for r in records if r["success"]]
    scenario_labels = [r["scenario"].replace(": ", "\n") for r in feasible_records]
    scenario_profits = [r["objective_solver"] for r in feasible_records]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(scenario_labels, scenario_profits)
    ax.set_ylabel("Прибыль, ден. ед.")
    ax.set_title("Сравнение оптимальной прибыли по сценариям")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, scenario_profits):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(scenario_profits) * 0.015,
                f"{value:,.0f}".replace(",", " "), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(scenario_profits) * 1.15)
    fig.tight_layout()
    fig.savefig(FIG / "scenario_comparison.png", dpi=180)
    plt.close(fig)

    # 4. Sensitivity ranges of objective coefficients.
    fig, ax = plt.subplots(figsize=(10, 6.2))
    products_sens = [row["Изделие"] for row in ranges]
    lower_vals = [row["Нижняя граница"] for row in ranges]
    upper_vals = [row["Верхняя граница"] for row in ranges]
    current_vals = [row["Текущая прибыль"] for row in ranges]
    y_pos = np.arange(len(products_sens))
    for i, (lo, hi, current) in enumerate(zip(lower_vals, upper_vals, current_vals)):
        left = 0.0 if lo is None else lo
        right = hi if hi is not None else current
        ax.hlines(i, left, right, linewidth=5)
        ax.plot(current, i, marker="o", markersize=6)
        if lo is None:
            ax.annotate("", xy=(0, i), xytext=(max(150, right * 0.07), i),
                        arrowprops={"arrowstyle": "<-", "lw": 1.4})
            ax.text(0, i - 0.18, "−∞", ha="left", va="bottom", fontsize=8)
        else:
            ax.text(lo, i - 0.18, f"{lo:.0f}", ha="center", va="bottom", fontsize=8)
        if hi is not None:
            ax.text(hi, i - 0.18, f"{hi:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(current, i + 0.18, f"текущая: {current:.0f}",
                ha="center", va="top", fontsize=8)
    ax.set_yticks(y_pos, labels=products_sens)
    ax.invert_yaxis()
    ax.set_ylim(len(products_sens) - 0.5, -0.8)
    ax.set_xlim(0, max(v for v in upper_vals if v is not None) * 1.18)
    ax.set_xlabel("Прибыль на единицу изделия, ден. ед./шт.")
    ax.set_title("Диапазоны устойчивости коэффициентов целевой функции")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity_ranges.png", dpi=180)
    plt.close(fig)

    # Human-readable solver log.
    lines = [
        "ЛАБОРАТОРНАЯ РАБОТА №1 — ВАРИАНТ 18",
        f"Seed: {data['seed']}",
        f"Python: {sys.version.split()[0]}",
        f"NumPy: {np.__version__}; pandas: {pd.__version__}; SciPy: {scipy.__version__}",
        "Solver: scipy.optimize.linprog(method='highs-ds', options={'presolve': True})",
        ""
    ]
    for rec in records:
        lines.extend([
            f"[{rec['scenario']}]",
            f"Status: {rec['status']} — {rec['solver_message']}",
            f"Objective: {rec['objective_solver']:.6f}",
            f"Independent check feasible: {rec['independent_check']['feasible']}",
            f"Recalculated objective: {rec['independent_check']['objective_recalculated']:.6f}",
            ""
        ])
    lines.extend([
        "[SENSITIVITY ANALYSIS]",
        "Optional assignment 4: objective coefficient stability ranges.",
        f"Basic products: {', '.join(sensitivity_meta['basic_products'])}",
        f"Active resources: {', '.join(sensitivity_meta['active_resources'])}"
    ])
    (OUT / "solver.log").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nBASE PRODUCTION:")
    print(products_df.to_string(index=False))
    print("\nRESOURCE ANALYSIS:")
    print(resources_df.to_string(index=False))
    print("\nSENSITIVITY RANGES:")
    print(pd.DataFrame(ranges).to_string(index=False))


if __name__ == "__main__":
    main()
