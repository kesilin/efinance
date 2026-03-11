# ========== 【核心：全局强制禁用代理，彻底解决翻墙冲突】 ==========
import os
# 清空所有系统代理环境变量，全系统兼容（Windows/Mac/Linux）
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''
os.environ['no_proxy'] = '*'  # 所有地址都不走代理

# 强制requests库不读取系统代理，所有网络请求直连
import requests
DEFAULT_SESSION = requests.Session()
DEFAULT_SESSION.trust_env = False  # 关键：禁用系统代理读取
# ========== 代理禁用结束 ==========

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# 屏蔽所有警告
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# 全局配置
CUMULATIVE_PERIOD = [7, 14, 21, 30, 60]
RISK_FREE_RATE = 0.03
C_FEE_RULE = {
    "申购费": 0,
    "持有<7天赎回费": 1.5,
    "持有≥7天赎回费": 0,
    "年销售服务费": 0.3
}

# ========== 工具函数 ==========
def format_num(num, decimals=4):
    """安全数值格式化"""
    if isinstance(num, (list, dict, np.ndarray)):
        return str(num)
    if pd.isna(num) or num is None:
        return "无数据"
    try:
        return f"{num:.{decimals}f}"
    except (ValueError, TypeError):
        return str(num)

def calculate_rsi(price_series, period=14):
    """RSI指标计算"""
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

def calculate_cumulative_performance(df, trade_days):
    """周期业绩计算"""
    df_recent = df.tail(trade_days).copy().reset_index(drop=True)
    if len(df_recent) < 2:
        return {"区间涨幅": "无数据", "最大回撤": "无数据", "夏普比率": "无数据", "卡玛比率": "无数据"}
    
    start_nav = df_recent['单位净值'].iloc[0]
    end_nav = df_recent['单位净值'].iloc[-1]
    total_return = (end_nav / start_nav - 1) * 100
    
    roll_max = df_recent['单位净值'].cummax()
    drawdown = (df_recent['单位净值'] / roll_max - 1) * 100
    max_drawdown = drawdown.min()
    
    daily_return = df_recent['单位净值'].pct_change().dropna()
    sharpe = "无数据"
    if len(daily_return) >= 20:
        annual_return = (total_return/100) / (len(df_recent)/252)
        annual_volatility = daily_return.std() * np.sqrt(252)
        if annual_volatility != 0:
            sharpe = round((annual_return - RISK_FREE_RATE) / annual_volatility, 4)
    
    calmar = "无数据"
    if max_drawdown < 0:
        calmar = round(total_return / abs(max_drawdown), 4)
    
    return {
        "区间涨幅": total_return,
        "最大回撤": max_drawdown,
        "夏普比率": sharpe,
        "卡玛比率": calmar
    }

def auto_get_benchmark(fund_type, fund_name):
    """自动匹配基准指数"""
    if "ETF联接" in fund_name or "ETF" in fund_type or "交易型开放式" in fund_type:
        if "科创板芯片" in fund_name:
            return "588290"
        elif "恒生科技" in fund_name:
            return "159742"
        elif "机器人" in fund_name:
            return "562500"
        elif "卫星通信" in fund_name:
            return "159579"
        elif "半导体" in fund_name or "芯片" in fund_name:
            return "588290"
        elif "黄金" in fund_name:
            return "AU9999.SGE"
    if "半导体" in fund_name or "芯片" in fund_name:
        return "990001"
    elif "新能源" in fund_name:
        return "399412"
    elif "军工" in fund_name:
        return "399967"
    elif "恒生科技" in fund_name:
        return "HSTECH.HK"
    elif "黄金" in fund_name:
        return "AU9999.SGE"
    elif "机器人" in fund_name:
        return "562500"
    return "000001"

def init_data_folder():
    """初始化数据文件夹，确保目录存在"""
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("基金波段分析报告"):
        os.makedirs("基金波段分析报告")
    
    # 初始化历史记录文件
    history_path = "data/history.csv"
    if not os.path.exists(history_path):
        pd.DataFrame(columns=[
            "基金代码", "基金名称", "最新净值", "最新分析时间", "净值更新日期"
        ]).to_csv(history_path, index=False, encoding="utf-8-sig")
    
    # 初始化持仓文件
    position_path = "data/position.csv"
    if not os.path.exists(position_path):
        pd.DataFrame(columns=[
            "基金代码", "基金名称", "持仓成本", "持仓份额", "买入日期", "最新净值", "浮盈浮亏", "浮盈浮亏比例"
        ]).to_csv(position_path, index=False, encoding="utf-8-sig")

def save_history(fund_code, fund_name, latest_nav, nav_date):
    """保存分析记录到历史文件"""
    history_path = "data/history.csv"
    df = pd.read_csv(history_path, encoding="utf-8-sig")
    # 确保基金代码为字符串格式（6位无前缀）
    fund_code = str(fund_code).zfill(6)
    # 已存在的记录更新，不存在新增
    if fund_code in df['基金代码'].astype(str).str.zfill(6).values:
        df.loc[df['基金代码'].astype(str).str.zfill(6) == fund_code, ["基金名称", "最新净值", "最新分析时间", "净值更新日期"]] = [
            fund_name, latest_nav, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), nav_date
        ]
    else:
        new_row = pd.DataFrame([{
            "基金代码": fund_code,
            "基金名称": fund_name,
            "最新净值": latest_nav,
            "最新分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "净值更新日期": nav_date
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(history_path, index=False, encoding="utf-8-sig")

def load_history(deduplicate=True):
    """加载历史分析记录
    
    Args:
        deduplicate: 是否按基金代码去重，保留最新的记录
    
    Returns:
        pd.DataFrame: 历史记录表
    """
    history_path = "data/history.csv"
    df = pd.read_csv(history_path, encoding="utf-8-sig")
    if not df.empty:
        # 确保基金代码为字符串格式（6位无前缀）
        df['基金代码'] = df['基金代码'].astype(str).str.zfill(6)
        if deduplicate:
            # 按基金代码分组，保留最新分析时间的记录
            df['最新分析时间'] = pd.to_datetime(df['最新分析时间'], errors='coerce')
            df = df.sort_values('最新分析时间', ascending=False).drop_duplicates('基金代码', keep='first')
            df = df.sort_values('最新分析时间', ascending=False).reset_index(drop=True)
    return df

def delete_history(fund_code):
    """删除历史记录"""
    history_path = "data/history.csv"
    df = pd.read_csv(history_path, encoding="utf-8-sig")
    # 确保基金代码为字符串格式
    fund_code = str(fund_code).zfill(6)
    df = df[df['基金代码'].astype(str).str.zfill(6) != fund_code]
    df.to_csv(history_path, index=False, encoding="utf-8-sig")