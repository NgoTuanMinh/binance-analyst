# scripts/auto_backtest_loop.py

import itertools
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Import các module đã có
from backtest.strategies.swing_strategy import SwingTradingStrategy
from backtest.engine import BacktestEngine


class AutoBacktestLoop:
    """
    Vòng lặp tự động chạy backtest với nhiều parameters
    Đọc features trực tiếp từ features/data/ (Parquet files)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = Path("backtest/auto_loop_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dữ liệu từ features
        self.features_dir = Path(config.get('features_dir', 'features/data'))
        self.symbols = config.get('symbols', ['BTCUSDT'])
        self.intervals = config.get('intervals', ['15m', '1h', '4h'])
        self.start_date = config.get('start_date', '2024-01-01')
        self.end_date = config.get('end_date', '2024-12-31')
        
        # Load dữ liệu
        self.data = self._load_features_data()
        
        # Kết quả
        self.results: List[Dict] = []
        self.best_result = None
        
    def _load_features_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load features từ Parquet files cho tất cả symbols
        """
        print("\n📂 Loading features data...")
        
        all_data = {}
        
        for symbol in self.symbols:
            # Tìm file Parquet cho symbol này
            parquet_files = list(self.features_dir.glob(f"{symbol}*.parquet"))
            
            if not parquet_files:
                print(f"⚠️ No features found for {symbol}, skipping...")
                continue
            
            # Load file đầu tiên (hoặc merge nếu nhiều)
            file_path = parquet_files[0]
            df = pd.read_parquet(file_path)
            
            # Đảm bảo có timestamp index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Filter theo date range
            df = df.loc[self.start_date:self.end_date] if self.start_date in df.index else df
            
            print(f"   ✅ {symbol}: {len(df)} rows, {len(df.columns)} features")
            all_data[symbol] = df
        
        if not all_data:
            raise ValueError(f"No data loaded from {self.features_dir}")
        
        return all_data
    
    def _get_feature_columns(self) -> List[str]:
        """
        Lấy danh sách feature columns (bỏ qua target columns)
        """
        sample_symbol = list(self.data.keys())[0]
        sample_df = self.data[sample_symbol]
        
        # Loại bỏ các cột target (có chứa 'target' trong tên)
        feature_cols = [col for col in sample_df.columns if 'target' not in col.lower()]
        
        return feature_cols
    
    def define_parameter_grid(self) -> Dict[str, List]:
        """
        Định nghĩa các parameters cần thử
        """
        return {
            # Strategy parameters
            'ema_period': [150, 200, 250],
            'rsi_period': [10, 14, 20],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'stop_loss_pct': [0.01, 0.015, 0.02],
            'take_profit_pct': [0.03, 0.045, 0.06],
            'max_hold_hours': [24, 48, 72],
            
            # ML filter parameters (optional)
            'use_ml_filter': [False],  # Có thể thêm [True, False]
            'ml_threshold': [0.65],
        }
    
    def generate_parameter_combinations(self) -> List[Dict]:
        """Tạo tất cả combinations của parameters"""
        param_grid = self.define_parameter_grid()
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            combinations.append(params)
        
        print(f"\n📊 Total parameter combinations: {len(combinations)}")
        return combinations
    
    def run_single_backtest(self, params: Dict, run_id: int) -> Dict:
        """
        Chạy 1 backtest với bộ parameters trên tất cả symbols
        Sử dụng dữ liệu đã load từ features
        """
        print(f"\n🏃 Running backtest {run_id}...")
        print(f"   Params: {params}")
        
        try:
            # Tổng hợp kết quả từ tất cả symbols
            all_trades = []
            all_metrics = []
            
            for symbol, df in self.data.items():
                # Prepare data cho backtest engine
                # Cần format đúng cấu trúc {interval: df}
                data_for_engine = self._prepare_data_for_engine(df, symbol)
                
                # Tạo strategy với params
                strategy = SwingTradingStrategy(params)
                
                # Chạy backtest
                engine = BacktestEngine(data_for_engine, strategy, params)
                results = engine.run()
                
                if results and results.get('trades'):
                    all_trades.extend(results['trades'])
                    all_metrics.append(results.get('metrics', {}))
            
            # Tổng hợp metrics
            if all_metrics:
                combined_metrics = self._aggregate_metrics(all_metrics, all_trades)
                
                return {
                    'run_id': run_id,
                    'params': params,
                    'metrics': combined_metrics,
                    'trades': all_trades,
                    'success': True
                }
            else:
                return {
                    'run_id': run_id,
                    'params': params,
                    'metrics': {},
                    'success': False,
                    'error': 'No trades generated'
                }
                
        except Exception as e:
            import traceback
            return {
                'run_id': run_id,
                'params': params,
                'metrics': {},
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _prepare_data_for_engine(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Chuẩn bị dữ liệu cho backtest engine
        Cần format: {interval: DataFrame}
        """
        # Đảm bảo có cột OHLCV cơ bản
        # Nếu features chỉ có indicators, cần tạo OHLCV từ đó
        
        if 'open' not in df.columns:
            # Tạo OHLCV từ features (nếu có)
            # Hoặc load từ data gốc
            print(f"⚠️ {symbol}: No OHLCV columns, using features only")
        
        # Tạm thời trả về dict với 1 interval
        return {'1h': df}
    
    def _aggregate_metrics(self, all_metrics: List[Dict], all_trades: List) -> Dict:
        """
        Tổng hợp metrics từ nhiều symbols
        """
        # Tính tổng số trades
        total_trades = len(all_trades)
        
        # Tính win rate tổng hợp
        winning_trades = [t for t in all_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Tính profit factor tổng hợp
        gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
        losing_trades = [t for t in all_trades if t.get('pnl', 0) <= 0]
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        
        # Tính avg return
        avg_return = np.mean([t.get('return_pct', 0) for t in all_trades]) if total_trades > 0 else 0
        
        # Tính Sharpe (đơn giản)
        returns = [t.get('return_pct', 0) for t in all_trades]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if returns else 0
        
        # Tính max drawdown (đơn giản)
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_return_per_trade': avg_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_drawdown),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
        }
    
    def run_all(self, max_combinations: int = None, parallel: bool = False):
        """
        Chạy tất cả combinations
        """
        combinations = self.generate_parameter_combinations()
        
        if max_combinations:
            combinations = combinations[:max_combinations]
        
        print(f"\n🚀 Starting auto backtest loop: {len(combinations)} combinations")
        print("="*60)
        
        start_time = time.time()
        
        for idx, params in enumerate(combinations, 1):
            result = self.run_single_backtest(params, idx)
            self.results.append(result)
            
            # In tiến độ
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (len(combinations) - idx)
            
            # In kết quả nhanh
            if result['success']:
                pf = result['metrics'].get('profit_factor', 0)
                wr = result['metrics'].get('win_rate', 0)
                print(f"   ✅ #{idx}: PF={pf:.3f} | WR={wr:.1%} | Trades={result['metrics'].get('total_trades', 0)}")
            else:
                print(f"   ❌ #{idx}: Failed - {result.get('error', 'Unknown')[:50]}")
            
            print(f"   Progress: {idx}/{len(combinations)} | "
                  f"Elapsed: {elapsed/60:.1f}m | "
                  f"ETA: {remaining/60:.1f}m")
        
        # Tìm kết quả tốt nhất
        self.find_best_result()
        
        # Lưu tất cả kết quả
        self.save_all_results()
        
        # Tạo báo cáo
        self.generate_report()
        
        return self.best_result
    
    def find_best_result(self):
        """Tìm bộ parameters tốt nhất (dựa trên profit factor)"""
        valid_results = [r for r in self.results if r['success']]
        
        if not valid_results:
            print("❌ No successful backtests")
            return
        
        # Sắp xếp theo profit factor
        valid_results.sort(
            key=lambda x: x['metrics'].get('profit_factor', 0),
            reverse=True
        )
        
        self.best_result = valid_results[0]
        
        print("\n" + "="*60)
        print("🏆 BEST RESULT FOUND")
        print("="*60)
        print(f"Profit Factor: {self.best_result['metrics'].get('profit_factor', 0):.4f}")
        print(f"Win Rate: {self.best_result['metrics'].get('win_rate', 0):.2%}")
        print(f"Avg Return: {self.best_result['metrics'].get('avg_return_per_trade', 0):.2%}")
        print(f"Total Trades: {self.best_result['metrics'].get('total_trades', 0)}")
        print(f"\nBest Parameters:")
        for k, v in self.best_result['params'].items():
            print(f"  {k}: {v}")
    
    def save_all_results(self):
        """Lưu tất cả kết quả"""
        # Lưu full results
        results_data = []
        for r in self.results:
            if r['success']:
                results_data.append({
                    'run_id': r['run_id'],
                    'profit_factor': r['metrics'].get('profit_factor', 0),
                    'win_rate': r['metrics'].get('win_rate', 0),
                    'avg_return': r['metrics'].get('avg_return_per_trade', 0),
                    'sharpe': r['metrics'].get('sharpe_ratio', 0),
                    'num_trades': r['metrics'].get('total_trades', 0),
                    **{f'param_{k}': v for k, v in r['params'].items()}
                })
        
        if results_data:
            df = pd.DataFrame(results_data)
            df.to_csv(self.results_dir / "all_results.csv", index=False)
            
            # Lưu best params
            if self.best_result:
                with open(self.results_dir / "best_params.json", 'w') as f:
                    json.dump(self.best_result['params'], f, indent=2)
    
    def generate_report(self):
        """Tạo báo cáo HTML"""
        if not self.results:
            return
        
        valid_results = [r for r in self.results if r['success']]
        
        if not valid_results:
            return
        
        df = pd.DataFrame([{
            'run_id': r['run_id'],
            'profit_factor': r['metrics'].get('profit_factor', 0),
            'win_rate': r['metrics'].get('win_rate', 0),
            'avg_return': r['metrics'].get('avg_return_per_trade', 0),
            'num_trades': r['metrics'].get('total_trades', 0)
        } for r in valid_results])
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Auto Backtest Loop Report</title>
            <style>
                body {{ font-family: Arial; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .best {{ background: #d4edda; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
                th {{ background: #2c3e50; color: white; }}
                .metric {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Auto Backtest Loop Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Symbols: {', '.join(self.symbols)}</p>
                <p>Period: {self.start_date} → {self.end_date}</p>
                <p>Total runs: {len(valid_results)} successful / {len(self.results)} total</p>
                
                <h2>📊 Top 10 Results by Profit Factor</h2>
                {df.nlargest(10, 'profit_factor').to_html(index=False, classes='table')}
                
                <h2>🎯 Best Parameters</h2>
                <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px;">{json.dumps(self.best_result['params'], indent=2) if self.best_result else 'None'}</pre>
                
                <h2>📈 Best Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Profit Factor</td><td class="metric">{self.best_result['metrics'].get('profit_factor', 0):.4f}</td></tr>
                    <tr><td>Win Rate</td><td>{self.best_result['metrics'].get('win_rate', 0):.2%}</td></tr>
                    <tr><td>Avg Return/Trade</td><td>{self.best_result['metrics'].get('avg_return_per_trade', 0):.2%}</td></tr>
                    <tr><td>Sharpe Ratio</td><td>{self.best_result['metrics'].get('sharpe_ratio', 0):.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td>{self.best_result['metrics'].get('max_drawdown', 0):.2%}</td></tr>
                    <tr><td>Total Trades</td><td>{self.best_result['metrics'].get('total_trades', 0)}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(self.results_dir / "report.html", 'w') as f:
            f.write(html)
        
        print(f"\n📄 Report saved to: {self.results_dir / 'report.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Backtest Loop with Features Data")
    
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"],
                        help="Symbols to test")
    
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="Start date (YYYY-MM-DD)")
    
    parser.add_argument("--end", type=str, default="2024-06-01",
                        help="End date (YYYY-MM-DD)")
    
    parser.add_argument("--intervals", nargs="+", default=["1h"],
                        help="Timeframes")
    
    parser.add_argument("--features-dir", type=str, default="features/data",
                        help="Directory containing Parquet features")
    
    parser.add_argument("--max-combinations", type=int, default=None,
                        help="Maximum combinations to test")
    
    parser.add_argument("--parallel", action="store_true",
                        help="Run in parallel (experimental)")
    
    args = parser.parse_args()
    
    config = {
        'symbols': args.symbols,
        'start_date': args.start,
        'end_date': args.end,
        'intervals': args.intervals,
        'features_dir': args.features_dir,
    }
    
    # Chạy vòng lặp
    loop = AutoBacktestLoop(config)
    best = loop.run_all(max_combinations=args.max_combinations)
    
    if best:
        print(f"\n✅ Optimization complete!")
        print(f"   Best profit factor: {best['metrics'].get('profit_factor', 0):.4f}")
        print(f"   Best params saved to: {loop.results_dir / 'best_params.json'}")
