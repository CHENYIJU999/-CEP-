#动态参数优化cep规则代码

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, Tuple, List, Any, Callable

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        try:
            self.df = pd.read_excel(self.file_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.sort_values(by=['symbol', 'date'], inplace=True)
            self.df.set_index('date', inplace=True)
            return self.df
        except Exception as e:
            print(f"数据加载错误: {e}")
            return pd.DataFrame()
    
    def add_common_features(self) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            raise ValueError("请先加载数据")
            
        grouped = self.df.groupby('symbol')
        
        # 添加基本特征
        self.df['is_up_day'] = self.df['close'] > self.df['open']
        self.df['prev_close'] = grouped['close'].transform(lambda x: x.shift(1))
        self.df['open_gap_pct'] = (self.df['open'] - self.df['prev_close']) / self.df['prev_close'] * 100
        
        return self.df


class RuleEvaluator:
    @staticmethod
    def evaluate_rule1(df: pd.DataFrame, volume_mult: float, fluctuation_thresh: float) -> float:
        """评估Rule1：高成交量大波动信号"""
        df = df.copy()
        grouped = df.groupby('symbol')
        df['vol_ma10'] = grouped['volume'].transform(lambda x: x.rolling(window=10, min_periods=10).mean())
        df['price_fluctuation'] = (df['high'] - df['low']) / df['open'] * 100
        df['Rule1'] = (df['volume'] > volume_mult * df['vol_ma10']) & (df['price_fluctuation'] > fluctuation_thresh)
        df['future_return'] = grouped['close'].transform(lambda x: x.shift(-3))
        df['future_return'] = (df['future_return'] - df['close']) / df['close']
        return df[df['Rule1']]['future_return'].mean() if df['Rule1'].sum() > 0 else -999
    
    @staticmethod
    def evaluate_rule2(df: pd.DataFrame, vol_increase_thresh: float) -> float:
        """评估Rule2：持续上涨 + 成交量上升"""
        df = df.copy()
        grouped = df.groupby('symbol')
        df['up_days_5'] = grouped['is_up_day'].transform(lambda x: x.rolling(window=5).sum())
        df['vol_ma5'] = grouped['volume'].transform(lambda x: x.rolling(window=5).mean())
        df['vol_ma5_prev'] = df['vol_ma5'].shift(5)
        df['vol_increase_pct'] = (df['vol_ma5'] - df['vol_ma5_prev']) / df['vol_ma5_prev'] * 100
        df['Rule2'] = (df['up_days_5'] == 5) & (df['vol_increase_pct'] > vol_increase_thresh)
        df['future_return'] = grouped['close'].transform(lambda x: x.shift(-3))
        df['future_return'] = (df['future_return'] - df['close']) / df['close']
        return df[df['Rule2']]['future_return'].mean() if df['Rule2'].sum() > 0 else -999
    
    @staticmethod
    def evaluate_rule3(df: pd.DataFrame, vol_ratio_thresh: float) -> float:
        """评估Rule3：价格突破 + 成交量放大"""
        df = df.copy()
        grouped = df.groupby('symbol')
        df['high_ma20'] = grouped['high'].transform(lambda x: x.rolling(window=20).max())
        df['vol_ma20'] = grouped['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['Rule3'] = (df['close'] > df['high_ma20']) & (df['volume'] > vol_ratio_thresh * df['vol_ma20'])
        df['future_return'] = grouped['close'].transform(lambda x: x.shift(-3))
        df['future_return'] = (df['future_return'] - df['close']) / df['close']
        return df[df['Rule3']]['future_return'].mean() if df['Rule3'].sum() > 0 else -999
    
    @staticmethod
    def evaluate_rule4(df: pd.DataFrame, std_drop_thresh: float, vol_drop_thresh: float) -> float:
        """评估Rule4：波动率与成交量双降"""
        df = df.copy()
        grouped = df.groupby('symbol')
        df['std7'] = grouped['close'].transform(lambda x: x.rolling(7).std())
        df['std7_prev'] = df['std7'].shift(7)
        df['std_drop'] = (df['std7_prev'] - df['std7']) / df['std7_prev'] * 100
        df['vol_ma7'] = grouped['volume'].transform(lambda x: x.rolling(7).mean())
        df['vol_ma7_prev'] = df['vol_ma7'].shift(7)
        df['vol_drop'] = (df['vol_ma7_prev'] - df['vol_ma7']) / df['vol_ma7_prev'] * 100
        df['Rule4'] = (df['std_drop'] > std_drop_thresh) & (df['vol_drop'] > vol_drop_thresh)
        df['future_return'] = grouped['close'].transform(lambda x: x.shift(-3))
        df['future_return'] = (df['future_return'] - df['close']) / df['close']
        return df[df['Rule4']]['future_return'].mean() if df['Rule4'].sum() > 0 else -999
    
    @staticmethod
    def evaluate_rule5(df: pd.DataFrame, open_gap_thresh: float, rule1_params: Tuple[float, float], rule3_params: float) -> float:
        """评估Rule5：Rule1 + Rule3 + 跳空高开"""
        df = df.copy()
        grouped = df.groupby('symbol')
        
        # 应用Rule1
        df['vol_ma10'] = grouped['volume'].transform(lambda x: x.rolling(window=10).mean())
        df['price_fluctuation'] = (df['high'] - df['low']) / df['open'] * 100
        rule1 = (df['volume'] > rule1_params[0] * df['vol_ma10']) & (df['price_fluctuation'] > rule1_params[1])
        
        # 应用Rule3
        df['high_ma20'] = grouped['high'].transform(lambda x: x.rolling(20).max())
        df['vol_ma20'] = grouped['volume'].transform(lambda x: x.rolling(20).mean())
        rule3 = (df['close'] > df['high_ma20']) & (df['volume'] > rule3_params * df['vol_ma20'])
        
        # 跳空高开
        df['Rule5'] = rule1 & rule3 & (df['open_gap_pct'] > open_gap_thresh)
        df['future_return'] = grouped['close'].transform(lambda x: x.shift(-3))
        df['future_return'] = (df['future_return'] - df['close']) / df['close']
        
        return df[df['Rule5']]['future_return'].mean() if df['Rule5'].sum() > 0 else -999


class GridSearchOptimizer:
    """网格搜索优化器"""
    def __init__(self, df: pd.DataFrame, evaluator: RuleEvaluator):
        self.df = df
        self.evaluator = evaluator
        
    def optimize(self) -> Dict[str, Any]:
        best_params = {}
        
        # Rule1优化
        print("正在优化Rule1参数...")
        rule1_space = product(np.arange(1.5, 3.1, 0.3), np.arange(2.0, 5.1, 0.5))
        best_params['Rule1'] = max(rule1_space, key=lambda p: self.evaluator.evaluate_rule1(self.df, p[0], p[1]))
        
        # Rule2优化
        print("正在优化Rule2参数...")
        rule2_space = np.arange(5, 25, 2)
        best_params['Rule2'] = max(rule2_space, key=lambda p: self.evaluator.evaluate_rule2(self.df, p))
        
        # Rule3优化
        print("正在优化Rule3参数...")
        rule3_space = np.arange(1.3, 2.5, 0.2)
        best_params['Rule3'] = max(rule3_space, key=lambda p: self.evaluator.evaluate_rule3(self.df, p))
        
        # Rule4优化
        print("正在优化Rule4参数...")
        rule4_space = product(np.arange(10, 35, 5), np.arange(20, 50, 5))
        best_params['Rule4'] = max(rule4_space, key=lambda p: self.evaluator.evaluate_rule4(self.df, p[0], p[1]))
        
        # Rule5优化 (依赖于Rule1/3的最优参数)
        print("正在优化Rule5参数...")
        rule5_space = np.arange(0.5, 3.0, 0.5)
        best_params['Rule5'] = max(rule5_space, 
                                   key=lambda p: self.evaluator.evaluate_rule5(self.df, p, best_params['Rule1'], best_params['Rule3']))
        
        return best_params


class RuleApplier:
    @staticmethod
    def apply_rules(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        grouped = df.groupby('symbol')
        
        # Rule1
        df['vol_ma10'] = grouped['volume'].transform(lambda x: x.rolling(window=10).mean())
        df['price_fluctuation'] = (df['high'] - df['low']) / df['open'] * 100
        df['Rule1_Complex'] = (df['volume'] > params['Rule1'][0] * df['vol_ma10']) & (df['price_fluctuation'] > params['Rule1'][1])
        
        # Rule2
        df['up_days_5'] = grouped['is_up_day'].transform(lambda x: x.rolling(window=5).sum())
        df['vol_ma5'] = grouped['volume'].transform(lambda x: x.rolling(window=5).mean())
        df['vol_ma5_prev'] = df['vol_ma5'].shift(5)
        df['vol_increase_pct'] = (df['vol_ma5'] - df['vol_ma5_prev']) / df['vol_ma5_prev'] * 100
        df['Rule2_TrendVol'] = (df['up_days_5'] == 5) & (df['vol_increase_pct'] > params['Rule2'])
        
        # Rule3
        df['high_ma20'] = grouped['high'].transform(lambda x: x.rolling(window=20).max())
        df['vol_ma20'] = grouped['volume'].transform(lambda x: x.rolling(window=20).mean())
        df['Rule3_Breakout'] = (df['close'] > df['high_ma20']) & (df['volume'] > params['Rule3'] * df['vol_ma20'])
        
        # Rule4
        df['std7'] = grouped['close'].transform(lambda x: x.rolling(7).std())
        df['std7_prev'] = df['std7'].shift(7)
        df['std_drop'] = (df['std7_prev'] - df['std7']) / df['std7_prev'] * 100
        df['vol_ma7'] = grouped['volume'].transform(lambda x: x.rolling(7).mean())
        df['vol_ma7_prev'] = df['vol_ma7'].shift(7)
        df['vol_drop'] = (df['vol_ma7_prev'] - df['vol_ma7']) / df['vol_ma7_prev'] * 100
        df['Rule4_VolatilityDrop'] = (df['std_drop'] > params['Rule4'][0]) & (df['vol_drop'] > params['Rule4'][1])
        
        # Rule5
        df['Rule5_ComplexConfluence'] = df['Rule1_Complex'] & df['Rule3_Breakout'] & (df['open_gap_pct'] > params['Rule5'])
        
        return df


class ResultProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def process_results(self) -> pd.DataFrame:
        rule_columns = ['Rule1_Complex', 'Rule2_TrendVol', 'Rule3_Breakout', 'Rule4_VolatilityDrop', 'Rule5_ComplexConfluence']
        self.df['matched_rules'] = ''
        
        for rule in rule_columns:
            self.df.loc[self.df[rule], 'matched_rules'] += rule + '; '
            
        self.df['matched_rules'] = self.df['matched_rules'].str.rstrip('; ')
        matches = self.df[self.df['matched_rules'] != ''].copy()
        
        # 中文名称映射
        rule_mapping = {
            'Rule1_Complex': '高成交量大波动信号',
            'Rule2_TrendVol': '持续上涨且成交量放大趋势',
            'Rule3_Breakout': '价格突破伴随成交量放大',
            'Rule4_VolatilityDrop': '波动率与成交量双降信号',
            'Rule5_ComplexConfluence': '高成交量突破确认信号'
        }
        
        matches['matched_rules_desc'] = matches['matched_rules'].apply(
            lambda x: '; '.join([rule_mapping.get(r, r) for r in x.split('; ')])
        )
        
        return matches
    
    def export_to_csv(self, matches: pd.DataFrame, output_path: str = 'rule_matches_optimized.csv') -> None:
        result = matches[['symbol', 'matched_rules_desc']].reset_index().rename(columns={
            'date': '时间', 'symbol': '股票名称', 'matched_rules_desc': '符合规则'
        })
        
        result.to_csv(output_path, index=False)
        print(f"结果已保存至 {output_path}")


def main():
    """主函数"""
    DATA_PATH = '/Users/chen/Desktop/stock/data-one year.xlsx'
    
    try:
        print("正在加载数据...")
        data_processor = DataProcessor(DATA_PATH)
        df = data_processor.load_data()
        df = data_processor.add_common_features()
        
        if df.empty:
            print("数据加载失败，程序退出")
            return
            
        print("开始参数优化...")
        evaluator = RuleEvaluator()
        optimizer = GridSearchOptimizer(df, evaluator)
        best_params = optimizer.optimize()
        
        print("\n最佳参数:")
        for rule, params in best_params.items():
            print(f"{rule}: {params}")

        print("\n正在应用优化规则...")
        rule_applier = RuleApplier()
        df_with_rules = rule_applier.apply_rules(df, best_params)

        print("\n正在处理结果...")
        result_processor = ResultProcessor(df_with_rules)
        matches = result_processor.process_results()

        print("\n部分匹配结果示例:")
        sample = matches[['symbol', 'matched_rules_desc']].reset_index().rename(columns={
            'date': '时间', 'symbol': '股票名称', 'matched_rules_desc': '符合规则'
        }).head()
        print(sample.to_string(index=False))
        result_processor.export_to_csv(matches)
        
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()
