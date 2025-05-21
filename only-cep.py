#没有优化的cep规则代码

import pandas as pd
import numpy as np

df = pd.read_excel('/Users/chen/Desktop/stock/data-one year.xlsx')
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['symbol', 'date'], inplace=True)
df.set_index('date', inplace=True)

preset_params = {
    'Rule1': (2.0, 3.0),       # (volume_mult, fluctuation_thresh)
    'Rule2': 15,               # vol_increase_thresh
    'Rule3': 1.8,              # vol_ratio_thresh
    'Rule4': (20, 30),         # (std_drop_thresh, vol_drop_thresh)
    'Rule5': 1.0               # open_gap_thresh
}

def apply_preset_rules(df, params):
    grouped = df.groupby('symbol')
    
    # Rule1 - 高成交量大波动信号
    df['vol_ma10'] = grouped['volume'].transform(lambda x: x.rolling(window=10, min_periods=10).mean())
    df['price_fluctuation'] = (df['high'] - df['low']) / df['open'] * 100
    df['Rule1_Complex'] = (df['volume'] > params['Rule1'][0] * df['vol_ma10']) & (df['price_fluctuation'] > params['Rule1'][1])

    # Rule2 - 持续上涨 + 成交量上升
    df['is_up_day'] = df['close'] > df['open']
    df['up_days_5'] = grouped['is_up_day'].transform(lambda x: x.rolling(window=5).sum())
    df['vol_ma5'] = grouped['volume'].transform(lambda x: x.rolling(window=5).mean())
    df['vol_ma5_prev'] = df['vol_ma5'].shift(5)
    df['vol_increase_pct'] = (df['vol_ma5'] - df['vol_ma5_prev']) / df['vol_ma5_prev'] * 100
    df['Rule2_TrendVol'] = (df['up_days_5'] == 5) & (df['vol_increase_pct'] > params['Rule2'])

    # Rule3 - 价格突破 + 成交量放大
    df['high_ma20'] = grouped['high'].transform(lambda x: x.rolling(window=20).max())
    df['vol_ma20'] = grouped['volume'].transform(lambda x: x.rolling(window=20).mean())
    df['Rule3_Breakout'] = (df['close'] > df['high_ma20']) & (df['volume'] > params['Rule3'] * df['vol_ma20'])

    # Rule4 - 波动率与成交量双降
    df['std7'] = grouped['close'].transform(lambda x: x.rolling(7).std())
    df['std7_prev'] = df['std7'].shift(7)
    df['std_drop'] = (df['std7_prev'] - df['std7']) / df['std7_prev'] * 100
    df['vol_ma7'] = grouped['volume'].transform(lambda x: x.rolling(7).mean())
    df['vol_ma7_prev'] = df['vol_ma7'].shift(7)
    df['vol_drop'] = (df['vol_ma7_prev'] - df['vol_ma7']) / df['vol_ma7_prev'] * 100
    df['Rule4_VolatilityDrop'] = (df['std_drop'] > params['Rule4'][0]) & (df['vol_drop'] > params['Rule4'][1])

    # Rule5 - Rule1 + Rule3 + 跳空高开
    df['prev_close'] = grouped['close'].transform(lambda x: x.shift(1))
    df['open_gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close'] * 100
    df['Rule5_ComplexConfluence'] = df['Rule1_Complex'] & df['Rule3_Breakout'] & (df['open_gap_pct'] > params['Rule5'])

    return df

print("应用预设规则...")
df = apply_preset_rules(df, preset_params)

rule_columns = ['Rule1_Complex', 'Rule2_TrendVol', 'Rule3_Breakout', 'Rule4_VolatilityDrop', 'Rule5_ComplexConfluence']
df['matched_rules'] = ''
for rule in rule_columns:
    df.loc[df[rule], 'matched_rules'] += rule + '; '
df['matched_rules'] = df['matched_rules'].str.rstrip('; ')
matches = df[df['matched_rules'] != ''].copy()

# 中文名称表
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

result = matches[['symbol', 'matched_rules_desc']].reset_index().rename(columns={
    'date': '时间', 'symbol': '股票名称', 'matched_rules_desc': '符合规则'
})
print(result.to_string(index=False))
result.to_csv('rule_matches_preset.csv', index=False)
print("已保存至 rule_matches_preset.csv")
