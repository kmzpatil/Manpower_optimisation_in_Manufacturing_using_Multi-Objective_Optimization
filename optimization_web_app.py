import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class ModelResult:
    model: str
    time_seconds: float
    iterations: int
    f1: float
    f2: float
    f3: float
    first_10_allocations: str
    allocations: np.ndarray


@dataclass
class OptimizationContext:
    n: int
    x_total: int
    seed: int
    z_value: float
    w1: float
    w2: float
    w3: float
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    bounds: list
    expected_output: np.ndarray
    output_variance: np.ndarray
    wage_costs: np.ndarray
    expected_idle_time: np.ndarray
    idle_variance: np.ndarray
    initial_x: np.ndarray
    f1_star: float
    f2_star: float
    f3_star: float


def variance_penalty(variance_array: np.ndarray, x_values: np.ndarray) -> float:
    # Vectorization keeps objective evaluation fast at higher dimensions.
    return float(np.sqrt(np.sum(variance_array * x_values**2)))


def robust_output_value(ctx: OptimizationContext, x_values: np.ndarray) -> float:
    return float(np.dot(ctx.expected_output, x_values) - ctx.z_value * variance_penalty(ctx.output_variance, x_values))


def wage_value(ctx: OptimizationContext, x_values: np.ndarray) -> float:
    return float(np.dot(ctx.wage_costs, x_values))


def robust_idle_value(ctx: OptimizationContext, x_values: np.ndarray) -> float:
    return float(np.dot(ctx.expected_idle_time, x_values) + ctx.z_value * variance_penalty(ctx.idle_variance, x_values))


def solve_single_objective(ctx: OptimizationContext, objective_func, x0: np.ndarray):
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - ctx.x_total}]
    return minimize(
        objective_func,
        x0,
        method="SLSQP",
        bounds=ctx.bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )


def normalize_weights(w1: float, w2: float, w3: float) -> tuple[float, float, float]:
    if min(w1, w2, w3) < 0:
        raise ValueError("Weights must be non-negative.")

    weight_sum = w1 + w2 + w3
    if weight_sum <= 0:
        raise ValueError("At least one weight must be greater than zero.")

    return w1 / weight_sum, w2 / weight_sum, w3 / weight_sum


@st.cache_data(show_spinner="Building simulation context...")
def build_context(n: int, x_total: int, seed: int, z_value: float, w1: float, w2: float, w3: float) -> OptimizationContext:
    if z_value < 0:
        raise ValueError("Z must be non-negative.")

    # Normalize to keep weighted deviations numerically stable and comparable across runs.
    w1, w2, w3 = normalize_weights(w1, w2, w3)

    rng = np.random.default_rng(seed)

    lower_bounds = np.full(n, 2.0)
    upper_bounds = np.full(n, 12.0)
    bounds = list(zip(lower_bounds, upper_bounds))

    expected_output = rng.integers(5, 26, size=n).astype(float)
    output_variance = rng.uniform(0.0001, 0.0009, size=n)
    wage_costs = rng.integers(10, 36, size=n).astype(float)
    expected_idle_time = rng.integers(20, 71, size=n).astype(float)
    idle_variance = rng.uniform(0.0001, 0.0009, size=n)

    initial_x = np.full(n, x_total / n)

    temp_ctx = OptimizationContext(
        n=n,
        x_total=x_total,
        seed=seed,
        z_value=z_value,
        w1=w1,
        w2=w2,
        w3=w3,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
        expected_output=expected_output,
        output_variance=output_variance,
        wage_costs=wage_costs,
        expected_idle_time=expected_idle_time,
        idle_variance=idle_variance,
        initial_x=initial_x,
        f1_star=0.0,
        f2_star=0.0,
        f3_star=0.0,
    )

    f1_star_result = solve_single_objective(temp_ctx, lambda x: -robust_output_value(temp_ctx, x), initial_x)
    f2_star_result = solve_single_objective(temp_ctx, lambda x: wage_value(temp_ctx, x), initial_x)
    f3_star_result = solve_single_objective(temp_ctx, lambda x: robust_idle_value(temp_ctx, x), initial_x)

    temp_ctx.f1_star = robust_output_value(temp_ctx, f1_star_result.x)
    temp_ctx.f2_star = wage_value(temp_ctx, f2_star_result.x)
    temp_ctx.f3_star = robust_idle_value(temp_ctx, f3_star_result.x)

    return temp_ctx


@st.cache_data(show_spinner="Computing manual utopia values...")
def build_context_manual(
    x_total: int,
    z_value: float,
    w1: float,
    w2: float,
    w3: float,
    manual_data_df: pd.DataFrame,
) -> OptimizationContext:
    lower_bounds = manual_data_df["Min Workers"].values
    upper_bounds = manual_data_df["Max Workers"].values
    expected_output = manual_data_df["Exp Output"].values
    output_variance = manual_data_df["Output Var"].values
    wage_costs = manual_data_df["Wage Cost"].values
    expected_idle_time = manual_data_df["Exp Idle"].values
    idle_variance = manual_data_df["Idle Var"].values
    if z_value < 0:
        raise ValueError("Z must be non-negative.")

    w1, w2, w3 = normalize_weights(w1, w2, w3)

    n = int(lower_bounds.shape[0])
    if n <= 0:
        raise ValueError("At least one work unit is required.")

    if np.any(lower_bounds > upper_bounds):
        raise ValueError("For each unit, minimum workers must be less than or equal to maximum workers.")

    if np.any(output_variance < 0) or np.any(idle_variance < 0):
        raise ValueError("Variance values must be non-negative.")

    if np.any(wage_costs < 0):
        raise ValueError("Wage costs must be non-negative.")

    min_total = float(np.sum(lower_bounds))
    max_total = float(np.sum(upper_bounds))
    if x_total < min_total or x_total > max_total:
        raise ValueError(
            f"Total manpower is infeasible for manual bounds. Valid range is {min_total:.0f} to {max_total:.0f}."
        )

    bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
    initial_x = np.clip(np.full(n, x_total / n), lower_bounds, upper_bounds)

    temp_ctx = OptimizationContext(
        n=n,
        x_total=x_total,
        seed=0,
        z_value=z_value,
        w1=w1,
        w2=w2,
        w3=w3,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
        expected_output=expected_output,
        output_variance=output_variance,
        wage_costs=wage_costs,
        expected_idle_time=expected_idle_time,
        idle_variance=idle_variance,
        initial_x=initial_x,
        f1_star=0.0,
        f2_star=0.0,
        f3_star=0.0,
    )

    f1_star_result = solve_single_objective(temp_ctx, lambda x: -robust_output_value(temp_ctx, x), initial_x)
    f2_star_result = solve_single_objective(temp_ctx, lambda x: wage_value(temp_ctx, x), initial_x)
    f3_star_result = solve_single_objective(temp_ctx, lambda x: robust_idle_value(temp_ctx, x), initial_x)

    if not (f1_star_result.success and f2_star_result.success and f3_star_result.success):
        raise RuntimeError("Unable to compute utopia values for manual input set.")

    temp_ctx.f1_star = robust_output_value(temp_ctx, f1_star_result.x)
    temp_ctx.f2_star = wage_value(temp_ctx, f2_star_result.x)
    temp_ctx.f3_star = robust_idle_value(temp_ctx, f3_star_result.x)

    return temp_ctx


def ccgc_terms(ctx: OptimizationContext, x_values: np.ndarray):
    output_term = (ctx.f1_star - np.dot(ctx.expected_output, x_values)) + ctx.z_value * variance_penalty(
        ctx.output_variance, x_values
    )
    wage_term = np.dot(ctx.wage_costs, x_values) - ctx.f2_star
    idle_term = (np.dot(ctx.expected_idle_time, x_values) - ctx.f3_star) + ctx.z_value * variance_penalty(
        ctx.idle_variance, x_values
    )
    return output_term, wage_term, idle_term


def solve_ccgc(ctx: OptimizationContext):
    x0 = ctx.initial_x
    initial_terms = ccgc_terms(ctx, x0)
    y0 = max(0.0, ctx.w1 * initial_terms[0], ctx.w2 * initial_terms[1], ctx.w3 * initial_terms[2])
    z0 = np.concatenate([x0, np.array([y0])])

    constraints = [
        {"type": "eq", "fun": lambda z: np.sum(z[:-1]) - ctx.x_total},
        {"type": "ineq", "fun": lambda z: z[-1] - ctx.w1 * ccgc_terms(ctx, z[:-1])[0]},
        {"type": "ineq", "fun": lambda z: z[-1] - ctx.w2 * ccgc_terms(ctx, z[:-1])[1]},
        {"type": "ineq", "fun": lambda z: z[-1] - ctx.w3 * ccgc_terms(ctx, z[:-1])[2]},
    ]

    return minimize(
        lambda z: z[-1],
        z0,
        method="SLSQP",
        bounds=ctx.bounds + [(0.0, None)],
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )


def cccp_terms(ctx: OptimizationContext, x_values: np.ndarray):
    output_term = (ctx.f1_star - np.dot(ctx.expected_output, x_values)) + ctx.z_value * variance_penalty(
        ctx.output_variance, x_values
    )
    wage_term = np.dot(ctx.wage_costs, x_values)
    idle_term = (np.dot(ctx.expected_idle_time, x_values) - ctx.f3_star) + ctx.z_value * variance_penalty(
        ctx.idle_variance, x_values
    )
    return output_term, wage_term, idle_term


def solve_cccp(ctx: OptimizationContext):
    x0 = ctx.initial_x
    output_term, wage_term, idle_term = cccp_terms(ctx, x0)
    epsilon1_0 = max(0.0, output_term)
    p1_0 = max(0.0, -output_term)
    n2_0 = max(0.0, wage_term - ctx.f2_star)
    epsilon3_0 = max(0.0, idle_term)
    p3_0 = max(0.0, -idle_term)
    y0 = max(0.0, ctx.w1 * (epsilon1_0 + p1_0), ctx.w2 * n2_0, ctx.w3 * (epsilon3_0 + p3_0))

    z0 = np.concatenate([x0, np.array([epsilon1_0, p1_0, n2_0, epsilon3_0, p3_0, y0])])

    def unpack(z_values):
        x_values = z_values[: ctx.n]
        epsilon1 = z_values[ctx.n]
        p1 = z_values[ctx.n + 1]
        n2 = z_values[ctx.n + 2]
        epsilon3 = z_values[ctx.n + 3]
        p3 = z_values[ctx.n + 4]
        y_value = z_values[ctx.n + 5]
        return x_values, epsilon1, p1, n2, epsilon3, p3, y_value

    constraints = [
        {"type": "eq", "fun": lambda z: np.sum(unpack(z)[0]) - ctx.x_total},
        {"type": "eq", "fun": lambda z: cccp_terms(ctx, unpack(z)[0])[0] - unpack(z)[1] + unpack(z)[2]},
        {"type": "eq", "fun": lambda z: cccp_terms(ctx, unpack(z)[0])[1] - unpack(z)[3] - ctx.f2_star},
        {"type": "eq", "fun": lambda z: cccp_terms(ctx, unpack(z)[0])[2] - unpack(z)[4] + unpack(z)[5]},
        {"type": "ineq", "fun": lambda z: unpack(z)[6] - ctx.w1 * (unpack(z)[1] + unpack(z)[2])},
        {"type": "ineq", "fun": lambda z: unpack(z)[6] - ctx.w2 * unpack(z)[3]},
        {"type": "ineq", "fun": lambda z: unpack(z)[6] - ctx.w3 * (unpack(z)[4] + unpack(z)[5])},
    ]

    variable_bounds = ctx.bounds + [(0.0, None)] * 6

    return minimize(
        lambda z: unpack(z)[6],
        z0,
        method="SLSQP",
        bounds=variable_bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )


def evaluate_solution(ctx: OptimizationContext, x_values: np.ndarray):
    f1 = (ctx.f1_star - np.dot(ctx.expected_output, x_values)) + ctx.z_value * variance_penalty(ctx.output_variance, x_values)
    f2 = np.dot(ctx.wage_costs, x_values) - ctx.f2_star
    f3 = (np.dot(ctx.expected_idle_time, x_values) - ctx.f3_star) + ctx.z_value * variance_penalty(ctx.idle_variance, x_values)
    return float(f1), float(f2), float(f3)


@st.cache_data(show_spinner="Executing solver logic...")
def run_model(ctx: OptimizationContext, model_name: str) -> ModelResult:
    start_time = time.perf_counter()

    if model_name == "CCGC":
        result = solve_ccgc(ctx)
        x_values = result.x[:-1]
    elif model_name == "CCCP":
        result = solve_cccp(ctx)
        x_values = result.x[: ctx.n]
    else:
        raise ValueError("Unsupported model selection.")

    if not result.success:
        raise RuntimeError(f"{model_name} optimization failed: {result.message}")

    elapsed = time.perf_counter() - start_time
    f1, f2, f3 = evaluate_solution(ctx, x_values)

    return ModelResult(
        model=model_name,
        time_seconds=elapsed,
        iterations=int(result.nit),
        f1=f1,
        f2=f2,
        f3=f3,
        first_10_allocations=np.round(x_values[:10], 4).tolist().__repr__(),
        allocations=x_values,
    )


def main():
    st.set_page_config(page_title="Manpower Optimization", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        .app-note {
            background: rgba(30, 41, 59, 0.05);
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(59, 130, 246, 0.2);
            backdrop-filter: blur(10px);
            color: #1e293b;
        }
        
        [data-theme="dark"] .app-note {
            background: rgba(255, 255, 255, 0.05);
            color: #f8fafc;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #3b82f6;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 0.1em;
        }

        .section-title::before {
            content: "";
            width: 3px;
            height: 16px;
            background: #3b82f6;
            border-radius: 2px;
        }

        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.4s ease !important;
            box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3) !important;
        }

        .stButton>button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.4) !important;
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        }

        [data-testid="stMetricValue"] {
            background: linear-gradient(90deg, #3b82f6, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Stochastic Manpower Optimization Dashboard")
    st.write(
        "Use this interface to compare CCGC and CCCP on the same synthetic manufacturing scenario. "
        "The app reports solver time, iteration count, objective values, and allocation similarity."
    )
    st.markdown(
        """
        <div class="app-note">
            <b>How to use:</b> choose Synthetic Benchmark or Manual Unit Data, set alpha and weights, then click Run Optimization.<br/>
            The page will show processing stages while solving and then render a formatted summary.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1.95], gap="large")

    with left:
        st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
        input_mode = st.selectbox("Input Mode", ["Synthetic Benchmark", "Manual Unit Data"], index=0)
        model = st.selectbox("Optimization Model", ["CCGC", "CCCP", "Both"], index=0)

        alpha = float(
            st.number_input(
                "Chance-Constraint Risk (alpha)",
                min_value=0.0001,
                max_value=0.9999,
                step=0.01,
                value=0.05,
                format="%.4f",
            )
        )
        z_value = float(norm.ppf(1.0 - alpha))
        st.caption(f"Computed Z from alpha: {z_value:.4f}")

        st.markdown("**Objective Weights**")
        w1 = float(st.number_input("w1 (Output)", min_value=0.0, step=0.05, value=0.4, format="%.3f"))
        w2 = float(st.number_input("w2 (Wage)", min_value=0.0, step=0.05, value=0.4, format="%.3f"))
        w3 = float(st.number_input("w3 (Idle)", min_value=0.0, step=0.05, value=0.2, format="%.3f"))
        raw_weight_sum = w1 + w2 + w3

        st.caption(f"Weight sum (raw): {raw_weight_sum:.4f} (normalized automatically at run time)")

        manual_data_df = None

        if input_mode == "Synthetic Benchmark":
            st.markdown("**System Constraints**")
            n = int(st.number_input("Work Units (N)", min_value=1, step=1, value=300))
            x_total = int(st.number_input("Total Manpower (X)", min_value=1, step=1, value=3000))
            seed = 42
            st.caption("Synthetic data seed is fixed to 42 for reproducibility.")
        else:
            st.markdown("**System Constraints & Unit Data**")
            n = int(st.number_input("Number of Work Units", min_value=1, step=1, value=5))
            x_total = int(st.number_input("Total Manpower", min_value=1, step=1, value=30))
            seed = 0

            st.caption("Edit the table below to adjust unit parameters. Changes are reactive.")
            
            # Default data for the table
            default_data = [
                {"Min Workers": 3.0, "Max Workers": 9.0, "Wage Cost": 20.0, "Exp Output": 8.0, "Output Var": 0.000125, "Exp Idle": 40.0, "Idle Var": 0.00016},
                {"Min Workers": 3.0, "Max Workers": 9.0, "Wage Cost": 15.0, "Exp Output": 6.0, "Output Var": 0.000324, "Exp Idle": 60.0, "Idle Var": 0.00021},
                {"Min Workers": 4.0, "Max Workers": 9.0, "Wage Cost": 17.0, "Exp Output": 10.0, "Output Var": 0.000469, "Exp Idle": 35.0, "Idle Var": 0.00013},
                {"Min Workers": 2.0, "Max Workers": 9.0, "Wage Cost": 12.0, "Exp Output": 6.0, "Output Var": 0.000521, "Exp Idle": 50.0, "Idle Var": 0.00022},
                {"Min Workers": 3.0, "Max Workers": 9.0, "Wage Cost": 18.0, "Exp Output": 10.0, "Output Var": 0.000324, "Exp Idle": 45.0, "Idle Var": 0.00019},
            ]
            
            # Ensure we have N rows
            if len(default_data) < n:
                for i in range(len(default_data), n):
                    default_data.append({
                        "Min Workers": 2.0, "Max Workers": 12.0, "Wage Cost": 15.0, 
                        "Exp Output": 8.0, "Output Var": 0.0002, "Exp Idle": 40.0, "Idle Var": 0.0002
                    })
            elif len(default_data) > n:
                default_data = default_data[:n]

            df_input = pd.DataFrame(default_data)
            edited_df = st.data_editor(
                df_input,
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    "Min Workers": st.column_config.NumberColumn("Min workers", help="Minimum required workers for this unit", format="%.1f", min_value=0.0),
                    "Max Workers": st.column_config.NumberColumn("Max workers", help="Maximum allowable workers for this unit", format="%.1f", min_value=0.0),
                    "Wage Cost": st.column_config.NumberColumn("Wage ($)", help="Cost per worker", format="$ %.1f", min_value=0.0),
                    "Exp Output": st.column_config.NumberColumn("Exp. Output", help="Expected production output", format="%.1f"),
                    "Output Var": st.column_config.NumberColumn("Output Var.", help="Variance in production output", format="%.6f"),
                    "Exp Idle": st.column_config.NumberColumn("Exp. Idle", help="Expected idle time", format="%.1f"),
                    "Idle Var": st.column_config.NumberColumn("Idle Var.", help="Variance in idle time", format="%.6f"),
                }
            )

            manual_data_df = edited_df
            st.caption(
                f"Manual feasibility range from bounds: {edited_df['Min Workers'].sum():.0f} to {edited_df['Max Workers'].sum():.0f}"
            )

        st.divider()
        run_clicked = st.button("Run Optimization", type="primary")

    with right:
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
        st.caption("Results appear below after optimization finishes.")
        if run_clicked:
            try:
                if input_mode == "Synthetic Benchmark":
                    ctx = build_context(n=n, x_total=x_total, seed=seed, z_value=z_value, w1=w1, w2=w2, w3=w3)
                else:
                    ctx = build_context_manual(
                        x_total=x_total,
                        z_value=z_value,
                        w1=w1,
                        w2=w2,
                        w3=w3,
                        manual_data_df=manual_data_df,
                    )

                st.markdown(
                    f"**Using normalized weights:** w1={ctx.w1:.4f}, w2={ctx.w2:.4f}, w3={ctx.w3:.4f} | **Z={ctx.z_value:.3f}**"
                )

                if model == "Both":
                    ccgc_result = run_model(ctx, "CCGC")
                    cccp_result = run_model(ctx, "CCCP")
                    results = [ccgc_result, cccp_result]
                    allocation_delta = float(np.max(np.abs(ccgc_result.allocations - cccp_result.allocations)))
                else:
                    results = [run_model(ctx, model)]
                    allocation_delta = None

                rows = [
                    {
                        "Model": r.model,
                        "Time (s)": f"{r.time_seconds:.6f}",
                        "Iterations": r.iterations,
                        "f1": f"{r.f1:.6f}",
                        "f2": f"{r.f2:.6f}",
                        "f3": f"{r.f3:.6f}",
                    }
                    for r in results
                ]

                st.success("Optimization completed successfully.")

                st.subheader("Optimization Summary")
                st.table(rows)

                metric_cols = st.columns(len(results))
                for idx, r in enumerate(results):
                    with metric_cols[idx]:
                        st.markdown(f"**{r.model}**")
                        st.metric("Time (s)", f"{r.time_seconds:.4f}")
                        st.metric("Iterations", f"{r.iterations}")
                        st.metric("f1", f"{r.f1:.4f}")
                        st.metric("f2", f"{r.f2:.4f}")
                        st.metric("f3", f"{r.f3:.4f}")

                for r in results:
                    with st.expander(f"{r.model} - First 10 Allocations", expanded=False):
                        st.code(r.first_10_allocations)

                if allocation_delta is not None:
                    st.markdown("**Allocation Difference (CCGC vs CCCP)**")
                    st.code(f"Max absolute difference: {allocation_delta:.8f}")

            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
