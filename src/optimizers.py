"""
Strategy Optimization Methods

This module provides three optimization methods for finding optimal
hyperparameters for trading strategies:

1. Random Search: Simple random sampling within bounds
2. Hyperopt (TPE): Tree-structured Parzen Estimator (Bayesian optimization)
3. Genetic Algorithm (DEAP): Evolutionary optimization

All methods work with the same strategy specification format and
return the best parameters found.
"""

import random
import copy
import warnings
from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from .backtester import run_backtest

warnings.filterwarnings("ignore")

# Optional: Hyperopt for Bayesian-like optimization
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    HYPEROPT_INSTALLED = True
except ImportError:
    HYPEROPT_INSTALLED = False

# Optional: DEAP for genetic algorithms
try:
    from deap import base, creator, tools, algorithms
    DEAP_INSTALLED = True
except ImportError:
    DEAP_INSTALLED = False


def calculate_average_yearly_gain(portfolio_values):
    """
    Calculate the average yearly percentage gain from portfolio values over time.
    
    Parameters
    ----------
    portfolio_values : list of dict
        List of dictionaries with 'date_time' (Timestamp) and 'value' keys.
    
    Returns
    -------
    float
        Average yearly percentage gain (e.g., 0.08 for 8%).
    """
    yearly_values = defaultdict(list)
    for record in portfolio_values:
        date = record["date_time"].to_pydatetime() if hasattr(record["date_time"], "to_pydatetime") else record["date_time"]
        yearly_values[date.year].append(record["value"])
    
    yearly_gains = []
    for year, values in sorted(yearly_values.items()):
        if len(values) >= 2:
            start_value = values[0]
            end_value = values[-1]
            yearly_gain = ((end_value - start_value) / start_value)
            yearly_gains.append(yearly_gain)
    
    if len(yearly_gains) > 0:
        average_yearly_gain = sum(yearly_gains) / len(yearly_gains)
    else:
        average_yearly_gain = 0.0

    return average_yearly_gain


def compile_backtest_results_sequential(results, data_frames):
    """
    Combine multiple backtest results into one dictionary.
    
    Preserves chronological order by shifting dates and scaling portfolio values
    to ensure continuity across multiple tickers or timeframes.
    
    Parameters
    ----------
    results : list of dict
        List of backtest result dictionaries.
    data_frames : list of DataFrame
        List of DataFrames used in the backtests.
    
    Returns
    -------
    dict
        Compiled results with adjusted portfolio values and chronological order.
    """
    first_backtest = copy.deepcopy(results[0])
    compiled_results = {}
        
    compiled_portfolio_values = first_backtest['portfolio_values_over_time']
    compiled_trades_history = first_backtest['trades_history']

    last_value = compiled_portfolio_values[-1]['value']
    last_date = pd.to_datetime(compiled_portfolio_values[-1]['date_time'])
    last_stock_value = compiled_portfolio_values[-1]['stock_value']

    for i in range(1, len(results)):
        current_results = copy.deepcopy(results[i])
        current_portfolio_values = copy.deepcopy(current_results['portfolio_values_over_time'])

        first_date = pd.to_datetime(current_portfolio_values[0]['date_time'])
        date_offset = last_date + timedelta(days=1) - first_date

        first_value = current_portfolio_values[0]['value']
        scaling_factor = last_value / first_value if first_value != 0 else 1

        first_stock_value = current_portfolio_values[0]['stock_value']
        stock_scaling_factor = last_stock_value / first_stock_value if first_stock_value != 0 else 1

        adjusted_portfolio_values = []
        for entry in current_portfolio_values:
            adjusted_value = entry['value'] * scaling_factor
            adjusted_stock_value = entry['stock_value'] * stock_scaling_factor
            adjusted_date = pd.to_datetime(entry['date_time']) + date_offset
            adjusted_portfolio_values.append({
                'date_time': adjusted_date,
                'value': adjusted_value,
                'stock_value': adjusted_stock_value
            })

        compiled_portfolio_values.extend(adjusted_portfolio_values)

        for trade in current_results['trades_history']:
            adjusted_trade = trade.copy()
            adjusted_trade['purchase_date'] += date_offset
            adjusted_trade['sale_date'] += date_offset
            compiled_trades_history.append(adjusted_trade)

        last_value = adjusted_portfolio_values[-1]['value']
        last_date = adjusted_portfolio_values[-1]['date_time']
        last_stock_value = adjusted_portfolio_values[-1]['stock_value']

    compiled_results['portfolio_values_over_time'] = compiled_portfolio_values
    compiled_results['trades_history'] = compiled_trades_history

    total_time_passed = compiled_portfolio_values[-1]['date_time'] - compiled_portfolio_values[0]['date_time']
    compiled_results['total_time_passed'] = total_time_passed
    
    total_years = total_time_passed / timedelta(days=365.25)
    compiled_results['total_years'] = total_years
    
    starting_value = compiled_portfolio_values[0]['value']
    ending_value = compiled_portfolio_values[-1]['value']
    compiled_results['total_percentage_gain'] = (ending_value / starting_value) - 1
    compiled_results['average_return_per_year'] = calculate_average_yearly_gain(compiled_portfolio_values)
    
    total_money_made = sum(r['total_amount_of_money_made'] for r in results)
    compiled_results['total_amount_of_money_made'] = total_money_made
    compiled_results['total_trades'] = len(compiled_trades_history)
    
    time_held_list = [trade['time_held'] for trade in compiled_trades_history]
    if time_held_list:
        average_time_holding_position = sum(time_held_list, timedelta()) / len(time_held_list)
    else:
        average_time_holding_position = timedelta(0)
    compiled_results['average_time_holding_position'] = average_time_holding_position
    compiled_results['average_trades_per_year'] = len(compiled_trades_history) / total_years if total_years > 0 else 0

    return compiled_results


def optimize(strategies, data_frames, loss_function, optimization_method='random', 
             max_evals=100, population_size=20, random_seed=None, verbose=True):
    """
    Optimize trading strategy hyperparameters using various methods.
    
    Parameters
    ----------
    strategies : list of dict
        Strategy specifications, each containing:
        - 'strategy': The strategy function
        - 'params': Base parameters (optional)
        - 'param_space': Hyperopt parameter space (for hyperopt method)
        - 'ga_bounds': Parameter bounds as {param_name: (min, max)}
    data_frames : list of DataFrame
        Data frames to backtest on (with 'ohlc' key containing OHLC data).
    loss_function : callable
        Takes backtest_results dict and returns float loss.
    optimization_method : str, default='random'
        One of 'random', 'hyperopt', or 'genetic'.
    max_evals : int, default=100
        Maximum evaluations/generations.
    population_size : int, default=20
        Population size for genetic algorithm.
    random_seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to show progress bars and output.
    
    Returns
    -------
    dict
        Results containing:
        - 'best_loss': Best loss value found
        - 'best_params': Best parameters found
        - 'best_strategy': Best strategy function
        - 'all_results': List of all evaluated (params, loss) pairs
    
    Raises
    ------
    ImportError
        If hyperopt or DEAP is not installed when required.
    ValueError
        If optimization_method is not recognized.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    best_loss = float('inf')
    best_params = None
    best_strategy = None
    all_results = []

    def evaluate_strategy(strategy, params):
        """Evaluate a strategy across all data frames."""
        if len(data_frames) > 1:
            results = []
            for df in data_frames:
                trading_signals = strategy(df['ohlc'], params)
                backtest_results, _ = run_backtest(trading_signals)
                results.append(backtest_results)
            combined_results = compile_backtest_results_sequential(results, data_frames)
            return loss_function(combined_results)
        else:
            trading_signals = strategy(data_frames[0]['ohlc'], params)
            backtest_results, _ = run_backtest(trading_signals)
            return loss_function(backtest_results)

    # 1. RANDOM SEARCH
    if optimization_method == 'random':
        eval_bar = tqdm(range(max_evals), desc="Random Search", disable=not verbose)
        
        for eval_num in eval_bar:
            for strategy_dict in strategies:
                strategy = strategy_dict['strategy']
                base_params = strategy_dict.get('params', {})
                bounds = strategy_dict.get('ga_bounds', {})
                
                current_params = {}
                for param, (min_val, max_val) in bounds.items():
                    current_params[param] = random.uniform(min_val, max_val)
                
                current_params.update({k: v for k, v in base_params.items() if k not in current_params})
                
                try:
                    loss = evaluate_strategy(strategy, current_params)
                    all_results.append({'params': current_params.copy(), 'loss': loss})
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_params = current_params.copy()
                        best_strategy = strategy
                except Exception as e:
                    if verbose:
                        tqdm.write(f"Evaluation failed: {e}")
                    continue
                
                if verbose:
                    eval_bar.set_postfix({
                        'Strategy': strategy.__name__,
                        'Loss': f"{loss:.4f}",
                        'Best': f"{best_loss:.4f}"
                    })

    # 2. HYPEROPT (Tree-structured Parzen Estimator)
    elif optimization_method == 'hyperopt':
        if not HYPEROPT_INSTALLED:
            raise ImportError("Hyperopt not installed. Install with 'pip install hyperopt'.")

        for strategy_dict in strategies:
            strategy = strategy_dict['strategy']
            param_space = strategy_dict['param_space']

            def objective(params):
                try:
                    current_loss = evaluate_strategy(strategy, params)
                    all_results.append({'params': params.copy(), 'loss': current_loss})
                    return {'loss': current_loss, 'status': STATUS_OK, 'params': params}
                except Exception as e:
                    return {'loss': float('inf'), 'status': STATUS_OK, 'params': params}

            trials = Trials()
            best_params_for_strategy = fmin(
                fn=objective,
                space=param_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                show_progressbar=verbose
            )
            final_loss = evaluate_strategy(strategy, best_params_for_strategy)

            if final_loss < best_loss:
                best_loss = final_loss
                best_params = best_params_for_strategy
                best_strategy = strategy

    # 3. GENETIC ALGORITHM (via DEAP)
    elif optimization_method == 'genetic':
        if not DEAP_INSTALLED:
            raise ImportError("DEAP not installed. Install with 'pip install deap'.")

        for strategy_dict in strategies:
            strategy = strategy_dict['strategy']
            bounds = strategy_dict['ga_bounds']
            
            # Create DEAP types (handle redefiniton)
            if hasattr(creator, "FitnessMin"):
                del creator.FitnessMin
            if hasattr(creator, "Individual"):
                del creator.Individual
            
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            toolbox = base.Toolbox()
            param_names = list(bounds.keys())
            
            for param_name in param_names:
                min_val, max_val = bounds[param_name]
                toolbox.register(f"attr_{param_name}", random.uniform, min_val, max_val)
            
            def safe_cxTwoPoint(ind1, ind2):
                if len(ind1) > 1 and len(ind2) > 1:
                    return tools.cxTwoPoint(ind1, ind2)
                return ind1, ind2
            
            toolbox.register("individual", tools.initCycle, creator.Individual,
                             [getattr(toolbox, f"attr_{name}") for name in param_names], n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            def evaluate(individual):
                param_dict = dict(zip(param_names, individual))
                try:
                    loss = evaluate_strategy(strategy, param_dict)
                    all_results.append({'params': param_dict.copy(), 'loss': loss})
                    return (loss,)
                except Exception:
                    return (float('inf'),)
            
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", safe_cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            pop = toolbox.population(n=population_size)
            
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", np.min)
            stats.register("avg", np.mean)
            
            result, logbook = algorithms.eaSimple(
                pop, toolbox,
                cxpb=0.7, mutpb=0.3,
                ngen=max_evals,
                stats=stats,
                verbose=verbose
            )
            
            best_ind = tools.selBest(pop, 1)[0]
            final_params = dict(zip(param_names, best_ind))
            final_loss = evaluate_strategy(strategy, final_params)
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = final_params
                best_strategy = strategy

    else:
        raise ValueError(f"Invalid optimization_method: {optimization_method}. Use 'random', 'hyperopt', or 'genetic'.")

    return {
        'best_loss': best_loss,
        'best_params': best_params,
        'best_strategy': best_strategy,
        'all_results': all_results
    }
