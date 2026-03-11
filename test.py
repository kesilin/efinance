import efinance as ef
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
#streamlit run app.py
# ==============================================
# 【唯一需要修改的地方】输入你要分析的基金6位代码
# 示例：020671 易方达科创板芯片C / 025209 永赢先锋半导体智选C
FUND_CODE = "562500" 
# ==============================================

# 全局配置
CUMULATIVE_PERIOD = [7, 14, 21, 30, 60]  # 累计分析周期
RISK_FREE_RATE = 0.03  # 年化无风险利率
C_FEE_RULE = {
    "申购费": 0,
    "持有<7天赎回费": 1.5,
    "持有≥7天赎回费": 0,
    "年销售服务费": 0.3
}
SAVE_FOLDER = "基金波段分析报告"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# 屏蔽所有警告
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# ========== 工具函数 ==========
def format_num(num, decimals=4):
    if isinstance(num, (list, dict, np.ndarray)):
        return str(num)
    if pd.isna(num) or num is None:
        return "无数据"
    try:
        return f"{num:.{decimals}f}"
    except (ValueError, TypeError):
        return str(num)

def calculate_rsi(price_series, period=14):
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

def calculate_cumulative_performance(df, trade_days):
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
    """自动匹配基金对应基准，适配主动/被动基金、次新基金"""
    if "ETF联接" in fund_name or "ETF" in fund_type:
        if "科创板芯片" in fund_name:
            return "588290"
        elif "恒生科技" in fund_name:
            return "159742"
        elif "卫星通信" in fund_name:
            return "159579"
        elif "半导体" in fund_name or "芯片" in fund_name:
            return "588290"
    if "半导体" in fund_name or "芯片" in fund_name:
        return "990001"
    elif "新能源" in fund_name:
        return "399412"
    elif "军工" in fund_name:
        return "399967"
    elif "恒生科技" in fund_name:
        return "HSTECH.HK"
    return "000001"

# ========== 通用基金分析类 ==========
class UniversalFundAnalyzer:
    def __init__(self, fund_code):
        self.fund_code = fund_code
        self.fund_name = "未知基金"
        self.fund_type = "混合型"
        self.benchmark_code = "000001"
        self.base_info = None
        self.history_nav = None
        self.position = None
        self.benchmark_nav = None
        self.gsz = "无数据"
        self.gsz_time = "无数据"
        self.beta = "无数据"
        self.alpha_annual = "无数据"
        self.tracking_error = "无数据"
        self._load_all_data()
    
    def _load_all_data(self):
        print(f"正在获取基金 {self.fund_code} 数据...")
        # 1. 自动获取基金基础信息【终极修复：基金类型全字段匹配】
        try:
            base_df = ef.fund.get_base_info([self.fund_code])
            self.base_info = base_df.iloc[0] if not base_df.empty else None
            if self.base_info is not None:
                self.fund_name = self.base_info.get('基金简称', '未知基金')
                # 全字段匹配基金类型，彻底解决未知类型问题
                for type_key in ['基金类型', '类型', '产品类型', '基金品类']:
                    if type_key in self.base_info and pd.notna(self.base_info[type_key]):
                        self.fund_type = self.base_info[type_key]
                        break
                print(f"成功识别：【{self.fund_name}】，基金类型：{self.fund_type}")
                # 自动匹配基准
                self.benchmark_code = auto_get_benchmark(self.fund_type, self.fund_name)
        except Exception as e:
            print(f"基金基础信息获取失败：{e}")
            self.base_info = None
        
        # 2. 【修复：实时盘中估值多接口兼容】
        try:
            # 优先用实时估值接口
            realtime_df = ef.fund.get_realtime_increase_rate(self.fund_code)
            if not realtime_df.empty:
                realtime_info = realtime_df.iloc[0]
                self.gsz = realtime_info.get('估算涨跌幅', '无数据')
                self.gsz_time = realtime_info.get('估算时间', '无数据')
            # 备用接口
            if self.gsz == "无数据" or self.gsz is None:
                realtime_df2 = ef.fund.get_realtime_quotes([self.fund_code])
                if not realtime_df2.empty:
                    realtime_info2 = realtime_df2.iloc[0]
                    self.gsz = realtime_info2.get('估算涨跌幅', '无数据')
                    self.gsz_time = realtime_info2.get('更新时间', '无数据')
            # 非交易时间填充收盘数据
            if self.gsz == "无数据" and self.base_info is not None:
                self.gsz = self.base_info.get('涨跌幅', '无数据')
                self.gsz_time = self.base_info.get('最新净值公开日期', '最新收盘日')
        except:
            if self.base_info is not None:
                self.gsz = self.base_info.get('涨跌幅', '无数据')
                self.gsz_time = self.base_info.get('最新净值公开日期', '最新收盘日')
        
        # 3. 历史净值数据
        try:
            self.history_nav = ef.fund.get_quote_history(self.fund_code)
            self.history_nav['日期'] = pd.to_datetime(self.history_nav['日期'])
            self.history_nav = self.history_nav.sort_values('日期').reset_index(drop=True)
        except:
            self.history_nav = pd.DataFrame()
            print("历史净值获取失败")
        
        # 4. 持仓数据
        try:
            self.position = ef.fund.get_invest_position(self.fund_code)
        except:
            self.position = pd.DataFrame()
        
        # 5. 基准指数数据
        try:
            self.benchmark_nav = ef.stock.get_quote_history(self.benchmark_code)
            if not self.benchmark_nav.empty:
                self.benchmark_nav['日期'] = pd.to_datetime(self.benchmark_nav['日期'])
                self.benchmark_nav = self.benchmark_nav.sort_values('日期').reset_index(drop=True)
                self.benchmark_nav = self.benchmark_nav.rename(columns={'收盘': '单位净值'})
        except:
            self.benchmark_nav = pd.DataFrame()
            print("基准数据获取失败，使用默认基准")
        
        # 6. Beta/Alpha/跟踪误差计算，适配次新基金
        if not self.history_nav.empty and not self.benchmark_nav.empty:
            merged_df = pd.merge(
                self.history_nav[['日期', '单位净值']],
                self.benchmark_nav[['日期', '单位净值']],
                on='日期', how='inner'
            ).dropna().sort_values('日期').reset_index(drop=True)
            
            if len(merged_df) >= 60:
                merged_df['基金日收益'] = merged_df['单位净值_x'].pct_change().fillna(0)
                merged_df['基准日收益'] = merged_df['单位净值_y'].pct_change().fillna(0)
                merged_df = merged_df[(merged_df['基金日收益'].abs() < 0.15) & (merged_df['基准日收益'].abs() < 0.15)]
                merged_df = merged_df.dropna()
                
                if len(merged_df) >= 40:
                    from scipy import stats
                    beta, alpha_daily, r_value, p_value, std_err = stats.linregress(
                        merged_df['基准日收益'],
                        merged_df['基金日收益']
                    )
                    self.beta = round(beta, 4)
                    self.alpha_annual = round(alpha_daily * 252 * 100, 4)
                    
                    tracking_diff = merged_df['基金日收益'] - merged_df['基准日收益']
                    self.tracking_error = round(tracking_diff.std() * np.sqrt(252) * 100, 4)
    
    def get_full_analysis(self):
        # 基础信息兜底
        latest_nav_date = self.history_nav['日期'].iloc[-1].strftime('%Y-%m-%d') if not self.history_nav.empty else '未知'
        latest_nav = self.history_nav['单位净值'].iloc[-1] if not self.history_nav.empty else '未知'
        
        base_info = {
            "基金代码": self.fund_code,
            "基金名称": self.fund_name,
            "基金类型": self.fund_type,
            "最新净值": latest_nav,
            "净值更新日期": latest_nav_date
        }
        if self.base_info is not None:
            base_info['基金公司'] = self.base_info.get('基金公司', '未知')
            base_info['成立日期'] = self.base_info.get('成立日期', '未知')
        
        # 最近7个交易日单日涨跌明细
        daily_detail = []
        if not self.history_nav.empty:
            recent_7d = self.history_nav.tail(7).sort_values('日期', ascending=False).reset_index(drop=True)
            for idx, row in recent_7d.iterrows():
                daily_detail.append({
                    "序号": f"第{idx+1}天",
                    "日期": row['日期'].strftime('%Y-%m-%d'),
                    "单位净值": row['单位净值'],
                    "单日涨跌幅": row['涨跌幅']
                })
        
        # 累计周期表现
        cumulative_perf = {}
        for days in CUMULATIVE_PERIOD:
            cumulative_perf[f"近{days}个交易日"] = calculate_cumulative_performance(self.history_nav, days)
        
        # 择时指标
        timing_indicator = {"择时指标": "无数据"}
        if not self.history_nav.empty and len(self.history_nav) >= 30:
            close_series = self.history_nav['单位净值']
            rsi_14 = calculate_rsi(close_series, 14)
            rolling_mean = close_series.rolling(20).mean()
            rolling_std = close_series.rolling(20).std()
            upper_band = rolling_mean.iloc[-1] + 2 * rolling_std.iloc[-1]
            lower_band = rolling_mean.iloc[-1] - 2 * rolling_std.iloc[-1]
            current_nav = close_series.iloc[-1]
            
            timing_indicator = {
                "RSI(14)": round(rsi_14, 4),
                "RSI状态": "超买（≥70，警惕回调）" if rsi_14 >=70 else "超卖（≤30，反弹机会）" if rsi_14 <=30 else "中性",
                "布林带上轨（压力位）": round(upper_band, 4),
                "当前净值": round(current_nav, 4),
                "布林带下轨（支撑位）": round(lower_band, 4),
                "交易成本说明": "持有≥7天，申购费+赎回费=0，无额外交易成本"
            }
        
        # 持仓分析
        position_analysis = {"持仓信息": "无数据"}
        if not self.position.empty:
            top10 = self.position.head(10)
            position_analysis = {
                "持仓公开日期": top10['公开日期'].iloc[0] if not top10.empty else "未知",
                "前10持仓总占比": round(top10['持仓占比'].sum(), 4) if not top10.empty else "无数据",
                "前3大重仓": top10[['股票简称', '持仓占比']].head(3).to_dict('records') if not top10.empty else []
            }
        
        # 基准对比
        benchmark_analysis = {"对比结果": "无数据"}
        if not self.benchmark_nav.empty and not self.history_nav.empty:
            fund_30d = calculate_cumulative_performance(self.history_nav, 30)["区间涨幅"]
            bench_30d = calculate_cumulative_performance(self.benchmark_nav, 30)["区间涨幅"]
            fund_total = calculate_cumulative_performance(self.history_nav, len(self.history_nav))["区间涨幅"]
            bench_total = calculate_cumulative_performance(self.benchmark_nav, len(self.history_nav))["区间涨幅"]
            
            rel_30d = fund_30d - bench_30d if (fund_30d != "无数据" and bench_30d != "无数据") else "无数据"
            rel_total = fund_total - bench_total if (fund_total != "无数据" and bench_total != "无数据") else "无数据"
            
            benchmark_analysis = {
                "跟踪基准": self.benchmark_code,
                "近30个交易日基金涨幅": fund_30d,
                "近30个交易日基准涨幅": bench_30d,
                "近30个交易日相对收益": rel_30d,
                "成立以来基金涨幅": fund_total,
                "成立以来基准涨幅": bench_total,
                "成立以来相对收益": rel_total,
                "Beta系数": self.beta,
                "年化Alpha": self.alpha_annual,
                "年化跟踪误差": self.tracking_error
            }
        
        return {
            "基金基础信息": base_info,
            "官方费率规则": C_FEE_RULE,
            "实时行情数据": {
                "估算时间": self.gsz_time,
                "估算涨跌幅": self.gsz,
                "最新收盘净值": latest_nav
            },
            "最近7个交易日单日涨跌明细": daily_detail,
            "波段周期累计表现": cumulative_perf,
            "波段择时指标": timing_indicator,
            "持仓结构分析": position_analysis,
            "基准对比分析": benchmark_analysis
        }

# ========== 生成Gemini提示词 ==========
def generate_gemini_prompt(analysis):
    bench_data = analysis.get('基准对比分析', {})
    rel_30d = bench_data.get('近30个交易日相对收益', '无数据')
    rel_total = bench_data.get('成立以来相对收益', '无数据')
    beta = bench_data.get('Beta系数', '无数据')
    te = bench_data.get('年化跟踪误差', '无数据')
    
    daily_detail_str = chr(10).join([
        f"- {item['序号']}({item['日期']}): 净值{format_num(item['单位净值'])}，单日涨跌幅{format_num(item['单日涨跌幅'])}%"
        for item in analysis['最近7个交易日单日涨跌明细']
    ])
    
    cumulative_str = chr(10).join([
        f"- {k}: 区间涨幅{format_num(v['区间涨幅'])}%，最大回撤{format_num(v['最大回撤'])}%，卡玛比率{format_num(v['卡玛比率'])}"
        for k, v in analysis['波段周期累计表现'].items()
    ])
    
    prompt = f"""
请你扮演我的场外基金波段交易专属顾问，基于以下数据，给出明确的7-30天波段操作建议，严格按照我的要求输出。

---
【核心规则（必须遵守）】
1.  该基金是场外C类基金，**持有≥7天赎回费0，持有<7天有1.5%惩罚性赎回费**，所有操作建议必须满足持有周期≥7天；
2.  我是波段投资者，持有周期严格锁定7-30天，单笔交易最大可承受亏损5%，目标收益5%-15%；
3.  所有建议必须结合数据，不能凭空预测，不能模棱两可。
---

【基金核心数据】
1.  基础信息：{analysis['基金基础信息']}
2.  实时行情：{analysis['实时行情数据']}
3.  最近7个交易日单日涨跌明细：
{daily_detail_str}
4.  波段周期累计表现：
{cumulative_str}
5.  波段择时指标：{analysis['波段择时指标']}
6.  持仓结构：{analysis['持仓结构分析']}
7.  基准对比：近30个交易日相对基准收益{format_num(rel_30d)}%，成立以来相对基准收益{format_num(rel_total)}%，Beta系数{format_num(beta)}，年化跟踪误差{format_num(te)}%

---
【我的需求】
请严格按以下3点输出，不要废话，不要额外内容：
1.  明确结论：现在是否适合申购做7-30天的波段？只能选「适合申购」/「暂时不建议申购」/「完全不建议申购」
2.  核心理由：不超过3条，必须结合最近7天的单日涨跌、累计周期表现、择时指标、赛道逻辑、交易成本
3.  波段操作计划：明确给出申购后的止盈位、止损位、建议持有周期、最佳赎回时间节点
    """
    return prompt

# ========== 打印分析报告 ==========
def print_analysis_report(analysis):
    fund_name = analysis['基金基础信息']['基金名称']
    fund_code = analysis['基金基础信息']['基金代码']
    print(f"\n======= {fund_name}({fund_code}) 7-30天波段分析报告 =======")
    
    for key in ["基金基础信息", "官方费率规则", "实时行情数据"]:
        print(f"\n【{key}】")
        value = analysis[key]
        for k, v in value.items():
            if key == "官方费率规则" and "服务费" in k:
                print(f"  - {k}: {v}%")
            else:
                val_str = format_num(v) if not isinstance(v, str) else v
                print(f"  - {k}: {val_str}")
    
    print(f"\n【最近7个交易日单日涨跌明细】")
    for item in analysis['最近7个交易日单日涨跌明细']:
        print(f"  - {item['序号']}({item['日期']}): 单位净值{format_num(item['单位净值'])}，单日涨跌幅{format_num(item['单日涨跌幅'])}%")
    
    print(f"\n【波段周期累计表现】")
    for k, v in analysis['波段周期累计表现'].items():
        print(f"  - {k}:")
        for sk, sv in v.items():
            val_str = format_num(sv) if not isinstance(sv, str) else sv
            print(f"    - {sk}: {val_str}")
    
    for key in ["波段择时指标", "持仓结构分析", "基准对比分析"]:
        print(f"\n【{key}】")
        value = analysis[key]
        for k, v in value.items():
            if isinstance(v, dict):
                print(f"  - {k}:")
                for sk, sv in v.items():
                    val_str = format_num(sv) if not isinstance(sv, str) else sv
                    print(f"    - {sk}: {val_str}")
            else:
                val_str = format_num(v) if not isinstance(v, str) else v
                print(f"  - {k}: {val_str}")

# ========== 执行主程序 ==========
if __name__ == "__main__":
    # 自动安装依赖
    try:
        from scipy import stats
    except ImportError:
        print("正在安装依赖scipy...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    
    # 初始化分析器
    analyzer = UniversalFundAnalyzer(FUND_CODE)
    analysis = analyzer.get_full_analysis()
    
    # 打印报告
    print_analysis_report(analysis)
    
    # 生成Gemini提示词
    gemini_prompt = generate_gemini_prompt(analysis)
    print("\n======= 给Gemini的波段操作提示词 =======")
    print(gemini_prompt)
    
    # 保存报告
    filename = f"{SAVE_FOLDER}/基金波段分析_{FUND_CODE}_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(gemini_prompt)
    print(f"\n分析报告已保存到：{filename}")