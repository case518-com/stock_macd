"""
台股技術分析回測系統
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 設定頁面
st.set_page_config(
    page_title="台股技術分析回測系統",
    page_icon="📈",
    layout="wide"
)

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataFetcher:
    """線上數據抓取類"""
    
    @staticmethod
    def fetch_stock_data(stock_id, start_date, end_date):
        """從 Yahoo Finance 抓取台股數據"""
        try:
            ticker = f"{stock_id}.TW"
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                ticker = f"{stock_id}.TWO"
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data[required_cols]
            
            return data
            
        except Exception as e:
            st.error(f"下載數據時發生錯誤: {e}")
            return None


class TechnicalIndicators:
    """技術指標計算類"""
    
    @staticmethod
    def calculate_ma(data, period):
        return data['Close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data, period):
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        ma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, ma, lower_band
    
    @staticmethod
    def calculate_kd(data, period=9, k_period=3, d_period=3):
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        rsv = 100 * (data['Close'] - low_min) / (high_max - low_min)
        k = rsv.ewm(span=k_period, adjust=False).mean()
        d = k.ewm(span=d_period, adjust=False).mean()
        return k, d


class BacktestEngine:
    """回測引擎"""
    
    def __init__(self, data, initial_capital=1000000):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = []
        
    def add_indicators(self):
        """計算所有技術指標"""
        ti = TechnicalIndicators()
        
        self.data['MA5'] = ti.calculate_ma(self.data, 5)
        self.data['MA20'] = ti.calculate_ma(self.data, 20)
        self.data['MA60'] = ti.calculate_ma(self.data, 60)
        self.data['RSI'] = ti.calculate_rsi(self.data, 14)
        
        macd, signal, histogram = ti.calculate_macd(self.data)
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = signal
        self.data['MACD_Histogram'] = histogram
        
        upper, middle, lower = ti.calculate_bollinger_bands(self.data)
        self.data['BB_Upper'] = upper
        self.data['BB_Middle'] = middle
        self.data['BB_Lower'] = lower
        
        k, d = ti.calculate_kd(self.data)
        self.data['K'] = k
        self.data['D'] = d
        
        return self.data
    
    def generate_signals(self, strategy='ma_cross'):
        """生成交易訊號（✅ 已加入 MACD > 0 過濾）"""
        self.data['Signal'] = 0
        
        if strategy == 'ma_cross':
            # 買進：MA5上穿MA20 且 MACD > 0
            self.data['Signal'] = np.where(
                (self.data['MA5'] > self.data['MA20']) & 
                (self.data['MA5'].shift(1) <= self.data['MA20'].shift(1)) &
                (self.data['MACD'] > 0), 1, 0  # ✅ 加入 MACD > 0
            )
            # 賣出：MA5下穿MA20
            self.data['Signal'] = np.where(
                (self.data['MA5'] < self.data['MA20']) & 
                (self.data['MA5'].shift(1) >= self.data['MA20'].shift(1)), -1, 
                self.data['Signal']
            )
        
        elif strategy == 'rsi':
            # 買進：RSI < 30 且 MACD > 0
            self.data['Signal'] = np.where(
                (self.data['RSI'] < 30) & 
                (self.data['RSI'].shift(1) >= 30) &
                (self.data['MACD'] > 0), 1, 0  # ✅ 加入 MACD > 0
            )
            # 賣出：RSI > 70
            self.data['Signal'] = np.where(
                (self.data['RSI'] > 70) & (self.data['RSI'].shift(1) <= 70), -1,
                self.data['Signal']
            )
        
        elif strategy == 'macd':
            # 買進：MACD上穿Signal 且 MACD > 0
            self.data['Signal'] = np.where(
                (self.data['MACD'] > self.data['MACD_Signal']) & 
                (self.data['MACD'].shift(1) <= self.data['MACD_Signal'].shift(1)) &
                (self.data['MACD'] > 0), 1, 0  # ✅ 加入 MACD > 0
            )
            # 賣出：MACD下穿Signal
            self.data['Signal'] = np.where(
                (self.data['MACD'] < self.data['MACD_Signal']) & 
                (self.data['MACD'].shift(1) >= self.data['MACD_Signal'].shift(1)), -1,
                self.data['Signal']
            )
        
        elif strategy == 'bollinger':
            # 買進：價格跌破下軌 且 MACD > 0
            self.data['Signal'] = np.where(
                (self.data['Close'] < self.data['BB_Lower']) & 
                (self.data['Close'].shift(1) >= self.data['BB_Lower'].shift(1)) &
                (self.data['MACD'] > 0), 1, 0  # ✅ 加入 MACD > 0
            )
            # 賣出：價格突破上軌
            self.data['Signal'] = np.where(
                (self.data['Close'] > self.data['BB_Upper']) & 
                (self.data['Close'].shift(1) <= self.data['BB_Upper'].shift(1)), -1,
                self.data['Signal']
            )
        
        elif strategy == 'kd':
            # 買進：K上穿D 且 K<20 且 MACD > 0
            self.data['Signal'] = np.where(
                (self.data['K'] > self.data['D']) & 
                (self.data['K'].shift(1) <= self.data['D'].shift(1)) & 
                (self.data['K'] < 20) &
                (self.data['MACD'] > 0), 1, 0  # ✅ 加入 MACD > 0
            )
            # 賣出：K下穿D 且 K>80
            self.data['Signal'] = np.where(
                (self.data['K'] < self.data['D']) & 
                (self.data['K'].shift(1) >= self.data['D'].shift(1)) & 
                (self.data['K'] > 80), -1,
                self.data['Signal']
            )
        
        return self.data
    
    def run_backtest(self, commission=0.001425, tax=0.003):
        """執行回測"""
        cash = self.initial_capital
        position = 0
        entry_price = 0
        
        for idx, row in self.data.iterrows():
            if row['Signal'] == 1 and position == 0:
                shares = int(cash / (row['Close'] * 1000)) * 1000
                if shares > 0:
                    cost = shares * row['Close'] * (1 + commission)
                    if cost <= cash:
                        cash -= cost
                        position = shares
                        entry_price = row['Close']
                        self.trades.append({
                            'Date': idx,
                            'Type': 'Buy',
                            'Price': row['Close'],
                            'Shares': shares,
                            'Cash': cash
                        })
            
            elif row['Signal'] == -1 and position > 0:
                proceeds = position * row['Close'] * (1 - commission - tax)
                cash += proceeds
                profit = (row['Close'] - entry_price) * position
                self.trades.append({
                    'Date': idx,
                    'Type': 'Sell',
                    'Price': row['Close'],
                    'Shares': position,
                    'Cash': cash,
                    'Profit': profit
                })
                position = 0
                entry_price = 0
            
            current_equity = cash + (position * row['Close'] if position > 0 else 0)
            self.equity_curve.append({
                'Date': idx,
                'Equity': current_equity,
                'Cash': cash,
                'Position': position
            })
        
        if position > 0:
            last_price = self.data.iloc[-1]['Close']
            proceeds = position * last_price * (1 - commission - tax)
            cash += proceeds
            profit = (last_price - entry_price) * position
            self.trades.append({
                'Date': self.data.index[-1],
                'Type': 'Sell (Final)',
                'Price': last_price,
                'Shares': position,
                'Cash': cash,
                'Profit': profit
            })
        
        return self.trades, self.equity_curve
    
    def calculate_metrics(self):
        """計算績效指標"""
        if len(self.equity_curve) == 0:
            return None
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        # 基本指標
        initial_capital = self.initial_capital
        final_capital = equity_df['Equity'].iloc[-1]
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        # 計算交易天數和年化報酬
        days = (equity_df['Date'].iloc[-1] - equity_df['Date'].iloc[0]).days
        years = days / 365.25
        annual_return = (((final_capital / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
        
        # 最大回撤
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak'] * 100
        max_drawdown = equity_df['Drawdown'].min()
        
        # 交易統計
        buy_trades = [t for t in self.trades if t['Type'] == 'Buy']
        sell_trades = [t for t in self.trades if 'Profit' in t]
        
        total_trades = len(sell_trades)
        winning_trades = len([t for t in sell_trades if t.get('Profit', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Sharpe Ratio
        equity_df['Returns'] = equity_df['Equity'].pct_change()
        sharpe_ratio = (equity_df['Returns'].mean() / equity_df['Returns'].std() * np.sqrt(252)) if equity_df['Returns'].std() != 0 else 0
        
        metrics = {
            '初始資金': initial_capital,
            '最終資金': final_capital,
            '總報酬率': total_return,
            '年化報酬率': annual_return,
            '最大回撤': max_drawdown,
            '總交易次數': total_trades,
            '勝率': win_rate,
            'Sharpe Ratio': sharpe_ratio
        }
        
        return metrics


def test_all_strategies(data, initial_capital=1000000):
    """✅ 測試所有策略（補上缺失的函式）"""
    strategies = ['ma_cross', 'rsi', 'macd', 'bollinger', 'kd']
    strategy_names = {
        'ma_cross': '均線交叉策略 (MA5/MA20)',
        'rsi': 'RSI策略',
        'macd': 'MACD策略',
        'bollinger': '布林通道策略',
        'kd': 'KD指標策略'
    }
    
    results = {}
    
    for strategy in strategies:
        engine = BacktestEngine(data, initial_capital)
        engine.add_indicators()
        engine.generate_signals(strategy)
        engine.run_backtest()
        metrics = engine.calculate_metrics()
        
        if metrics:
            results[strategy_names[strategy]] = {
                'metrics': metrics,
                'data': engine.data,
                'trades': engine.trades,
                'equity_curve': engine.equity_curve
            }
    
    return results


def plot_chart(data, equity_curve, trades, title):
    """繪製技術分析圖表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子圖1：價格與均線
    ax1.plot(data.index, data['Close'], label='收盤價', linewidth=1.5, color='black')
    ax1.plot(data.index, data['MA5'], label='MA5', linewidth=1, alpha=0.7, color='blue')
    ax1.plot(data.index, data['MA20'], label='MA20', linewidth=1, alpha=0.7, color='red')
    ax1.plot(data.index, data['MA60'], label='MA60', linewidth=1, alpha=0.7, color='green')
    
    # 標記買賣點
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    ax1.scatter(buy_signals.index, buy_signals['Close'], color='red', marker='^', s=100, label='買進', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['Close'], color='green', marker='v', s=100, label='賣出', zorder=5)
    
    ax1.set_title(f'{title} - 價格與均線')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('價格 (元)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 子圖2：MACD
    ax2.plot(data.index, data['MACD'], label='MACD', linewidth=1.5, color='blue')
    ax2.plot(data.index, data['MACD_Signal'], label='Signal', linewidth=1.5, color='red')
    ax2.bar(data.index, data['MACD_Histogram'], label='Histogram', color='gray', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_title('MACD指標')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('MACD')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 子圖3：KD指標
    ax3.plot(data.index, data['K'], label='K', linewidth=1.5, color='blue')
    ax3.plot(data.index, data['D'], label='D', linewidth=1.5, color='red')
    ax3.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='超買(80)')
    ax3.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='超賣(20)')
    ax3.fill_between(data.index, 0, 20, color='green', alpha=0.1)
    ax3.fill_between(data.index, 80, 100, color='red', alpha=0.1)
    ax3.set_title('KD指標')
    ax3.set_xlabel('日期')
    ax3.set_ylabel('KD值')
    ax3.set_ylim([0, 100])
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 子圖4：權益曲線
    equity_df = pd.DataFrame(equity_curve)
    ax4.plot(equity_df['Date'], equity_df['Equity'], label='總權益', linewidth=2, color='darkblue')
    ax4.axhline(y=equity_df['Equity'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='初始資金')
    ax4.fill_between(equity_df['Date'], equity_df['Equity'].iloc[0], equity_df['Equity'], 
                    where=equity_df['Equity'] >= equity_df['Equity'].iloc[0], 
                    color='green', alpha=0.3)
    ax4.fill_between(equity_df['Date'], equity_df['Equity'].iloc[0], equity_df['Equity'], 
                    where=equity_df['Equity'] < equity_df['Equity'].iloc[0], 
                    color='red', alpha=0.3)
    ax4.set_title('權益曲線')
    ax4.set_xlabel('日期')
    ax4.set_ylabel('權益 (元)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Streamlit 主程式
def main():
    st.title("📈 台股技術分析回測系統（MACD 多頭過濾版）")
    st.markdown("✅ 所有買進訊號都需 MACD > 0（多頭趨勢）- 已修正並真正實作！")
    st.markdown("---")
    
    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 回測設定")
        
        # 股票代號
        stock_id = st.text_input("股票代號", value="2330", help="例如：2330 (台積電)")
        
        # 時間範圍
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "起始日期",
                value=datetime(2025, 1, 1),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "結束日期",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # 初始資金
        initial_capital = st.number_input(
            "初始資金 (元)",
            min_value=100000,
            max_value=100000000,
            value=1000000,
            step=100000
        )
        
        st.markdown("---")
        st.info("💡 系統會自動測試所有策略並顯示最佳結果")
        st.success("✅ 所有買進都需 MACD > 0（已真正實作）")
        
        # 執行回測按鈕
        run_backtest = st.button("🚀 執行回測", type="primary", use_container_width=True)
    
    # 主要內容區
    if run_backtest:
        if not stock_id:
            st.error("請輸入股票代號")
            return
        
        # 顯示進度
        with st.spinner(f"正在下載 {stock_id} 的數據..."):
            data = DataFetcher.fetch_stock_data(stock_id, start_date, end_date)
        
        if data is None or data.empty:
            st.error(f"❌ 無法下載 {stock_id} 的數據，請檢查股票代號是否正確")
            return
        
        st.success(f"✓ 成功下載 {len(data)} 筆數據 ({data.index[0].date()} 至 {data.index[-1].date()})")
        
        # 測試所有策略
        with st.spinner("正在測試所有策略..."):
            all_results = test_all_strategies(data, initial_capital)
        
        if not all_results:
            st.error("❌ 無法生成回測結果")
            return
        
        # 找出最佳策略
        best_strategy_name = max(all_results.items(), 
                                key=lambda x: x[1]['metrics']['總報酬率'])
        
        best_strategy = best_strategy_name[0]
        best_result = best_strategy_name[1]
        metrics = best_result['metrics']
        
        # 顯示最佳策略標題
        st.markdown(f"## 🏆 最佳策略：{best_strategy}")
        
        # 顯示績效指標
        st.markdown("### 📊 績效指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "總報酬率",
                f"{metrics['總報酬率']:.2f}%",
                delta=f"{metrics['總報酬率']:.2f}%"
            )
        
        with col2:
            st.metric("年化報酬率", f"{metrics['年化報酬率']:.2f}%")
        
        with col3:
            st.metric("勝率", f"{metrics['勝率']:.2f}%")
        
        with col4:
            st.metric("最大回撤", f"{metrics['最大回撤']:.2f}%")
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("初始資金", f"${metrics['初始資金']:,.0f}")
        
        with col6:
            profit = metrics['最終資金'] - metrics['初始資金']
            st.metric("最終資金", f"${metrics['最終資金']:,.0f}", delta=f"${profit:,.0f}")
        
        with col7:
            st.metric("總交易次數", f"{metrics['總交易次數']}")
        
        with col8:
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        
        # 顯示所有策略比較表
        st.markdown("---")
        st.markdown("### 📊 所有策略比較")
        
        comparison_data = []
        for strategy_name, result in all_results.items():
            m = result['metrics']
            comparison_data.append({
                '策略': strategy_name,
                '總報酬率': f"{m['總報酬率']:.2f}%",
                '年化報酬率': f"{m['年化報酬率']:.2f}%",
                '勝率': f"{m['勝率']:.2f}%",
                '最大回撤': f"{m['最大回撤']:.2f}%",
                '交易次數': m['總交易次數'],
                'Sharpe Ratio': f"{m['Sharpe Ratio']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('總報酬率', ascending=False)
        st.dataframe(comparison_df, use_container_width=True)
        
        # 顯示圖表
        st.markdown("---")
        st.markdown("### 📈 技術分析圖表")
        
        fig = plot_chart(
            best_result['data'], 
            best_result['equity_curve'], 
            best_result['trades'], 
            f"{stock_id} ({best_strategy})"
        )
        st.pyplot(fig)
        
        # 顯示交易明細
        st.markdown("---")
        st.markdown("### 📋 交易明細")
        
        trades = best_result['trades']
        if len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            trades_df['Date'] = pd.to_datetime(trades_df['Date']).dt.date
            st.dataframe(trades_df, use_container_width=True)
            
            # 下載CSV
            csv = trades_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下載交易紀錄 (CSV)",
                data=csv,
                file_name=f"{stock_id}_{best_strategy}_trades_log.csv",
                mime="text/csv"
            )
        else:
            st.info("此策略在選定期間內沒有產生任何交易訊號")
    
    else:
        # 初始畫面
        st.markdown("""
        ### 👋 歡迎使用台股技術分析回測系統！
        
        #### 📌 使用說明：
        1. 在左側輸入**股票代號**（例如：2330）
        2. 設定**回測時間範圍**
        3. 點擊「🚀 執行回測」開始分析
        4. 系統會自動測試所有策略並顯示**最佳結果**
        
        #### 🔒 MACD 多頭過濾機制（✅ 已修正）：
        - ✅ **所有買進訊號都需 MACD > 0**
        - ✅ 只在多頭趨勢中操作
        - ✅ 避免空頭市場的逆勢交易
        - ✅ 提高勝率、降低風險
        
        #### 📊 自動測試的策略：
        - ✅ 均線交叉策略（+ MACD > 0）
        - ✅ RSI策略（+ MACD > 0）
        - ✅ MACD策略（+ MACD > 0）
        - ✅ 布林通道策略（+ MACD > 0）
        - ✅ KD策略（+ MACD > 0）
        
        #### 💡 特色：
        - 🚀 **自動測試**所有策略
        - 🏆 **智慧推薦**最佳策略
        - 📊 **完整比較**所有策略績效
        - 🔒 **多頭過濾**只做順勢交易
        - 📈 **視覺化**技術分析圖表
        """)
        
        # 顯示熱門股票
        st.markdown("---")
        st.markdown("### 🔥 熱門股票代號參考")
        
        popular_stocks = {
            "台積電": "2330",
            "鴻海": "2317",
            "聯發科": "2454",
            "中華電": "2412",
            "富邦金": "2881",
            "國泰金": "2882",
            "台達電": "2308",
            "聯電": "2303"
        }
        
        cols = st.columns(4)
        for idx, (name, code) in enumerate(popular_stocks.items()):
            with cols[idx % 4]:
                st.button(f"{name} ({code})", key=code, use_container_width=True)


if __name__ == "__main__":
    main()
