# Libo-bot
# ai_trading_bot_pro.py
"""
AI Trading Bot Pro - Starter
- supports backtest and live (paper) trading modes
- modular: Data -> Features -> Model -> Risk -> Execution -> Supervisor
"""

import time
import logging
import threading
import sqlite3
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# External libs: ccxt, sklearn, ta
import ccxt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import ta

# ---------- CONFIG ----------
CONFIG = {
    "exchange": "binance",
    "symbol": "BTC/USDT",
    "timeframe": "5m",
    "mode": "paper",  # "backtest", "paper", "live"
    "risk": {
        "risk_per_trade_pct": 0.5,      # percent of account equity to risk per trade
        "max_drawdown_pct": 10.0,       # stop trading if reached
        "max_position_pct": 20.0,       # max % of account equity in one position
        "stop_loss_pct": 1.5,           # per trade stop-loss percent
    },
    "paper_balance": 10000.0,           # used in paper/backtest
    "telegram": {
        "bot_token": "",    # fill for alerts
        "chat_id": "",      # fill for alerts
    },
    "logging_level": "INFO",
    "db": "bot_state.db",
}

# ---------- LOGGING ----------
logging.basicConfig(level=CONFIG["logging_level"])
logger = logging.getLogger("ai-trader")

# ---------- STATE STORAGE ----------
def init_db():
    conn = sqlite3.connect(CONFIG["db"])
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER,
        symbol TEXT,
        side TEXT,
        size REAL,
        price REAL,
        pnl REAL,
        note TEXT
    )
    """)
    conn.commit()
    conn.close()

# ---------- ALERTING ----------
import requests
def telegram_alert(text: str):
    token = CONFIG["telegram"]["bot_token"]
    chat_id = CONFIG["telegram"]["chat_id"]
    if not token or not chat_id:
        logger.debug("Telegram creds empty, skip alert: %s", text)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as e:
        logger.exception("Failed to send telegram alert: %s", e)

# ---------- DATA INGESTION ----------
def fetch_ohlcv(exchange, symbol, timeframe, limit=500):
    # ccxt fetchOHLCV
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def ohlcv_to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

# ---------- FEATURES ----------
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["ma10"] = ta.trend.sma_indicator(df["close"], window=10)
    df["ma50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["vol_ema"] = ta.trend.ema_indicator(df["vol"], window=20)
    df = df.dropna()
    return df

# ---------- MODEL (example) ----------
@dataclass
class ModelWrapper:
    scaler: Optional[StandardScaler] = None
    clf: Optional[LogisticRegression] = None

    def train(self, df: pd.DataFrame):
        # Create simple target: next-close > close by small margin
        df = df.copy()
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        features = ["rsi","ma10","ma50","vol_ema"]
        X = df[features].iloc[:-1]
        y = df["target"].iloc[:-1]
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.clf = LogisticRegression(max_iter=200).fit(Xs, y)
        logger.info("Model trained on %d samples", len(X))

    def predict(self, row: pd.Series) -> Dict[str, Any]:
        # returns signal and confidence
        features = ["rsi","ma10","ma50","vol_ema"]
        X = np.array(row[features]).reshape(1, -1)
        Xs = self.scaler.transform(X)
        prob = self.clf.predict_proba(Xs)[0,1]
        signal = "buy" if prob > 0.55 else ("sell" if prob < 0.45 else "hold")
        return {"signal": signal, "confidence": float(prob)}

# ---------- RISK MANAGER ----------
class RiskManager:
    def __init__(self, config):
        self.cfg = config["risk"]

    def compute_size(self, equity, price) -> float:
        # fixed fractional position sizing
        risk_amount = equity * (self.cfg["risk_per_trade_pct"]/100.0)
        # assuming stop_loss_pct defines how big a move will be stopped out at:
        stop_pct = self.cfg["stop_loss_pct"]/100.0
        if stop_pct <= 0:
            return 0.0
        size = risk_amount / (price * stop_pct)
        max_pos_value = equity * (self.cfg["max_position_pct"]/100.0)
        if size * price > max_pos_value:
            size = max_pos_value / price
        return float(round(size, 6))

# ---------- EXECUTOR ----------
class Executor:
    def __init__(self, exchange, symbol, risk_mgr, paper_balance=None):
        self.exchange = exchange
        self.symbol = symbol
        self.risk_mgr = risk_mgr
        self.paper_balance = paper_balance
        self.positions = []  # simplified

    def get_equity(self):
        if CONFIG["mode"] == "paper":
            return self.paper_balance
        else:
            # Query exchange balance (example)
            bal = self.exchange.fetch_balance()
            # naive: use USDT balance as equity
            return float(bal.get("USDT", {}).get("total", 0.0))

    def place_order(self, side, size, price=None):
        logger.info("Placing order %s %s @ %s", side, size, price)
        if CONFIG["mode"] == "paper":
            # simulate fill at price
            fill_price = price
            pnl = 0.0
            ts = int(time.time())
            conn = sqlite3.connect(CONFIG["db"])
            cur = conn.cursor()
            cur.execute("INSERT INTO trades (ts,symbol,side,size,price,pnl,note) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (ts, self.symbol, side, size, fill_price, pnl, "paper_fill"))
            conn.commit(); conn.close()
            # update paper balance if closing sell/buy etc. Simplified: if buy -> reduce balance by size*price
            if side == "buy":
                self.paper_balance -= size * fill_price
            else:
                self.paper_balance += size * fill_price
            return {"status":"filled", "price": fill_price}
        else:
            # live trading (use market order for the prototype)
            params = {}
            try:
                if side == "buy":
                    order = self.exchange.create_market_buy_order(self.symbol, size, params)
                else:
                    order = self.exchange.create_market_sell_order(self.symbol, size, params)
                return order
            except Exception as exc:
                logger.exception("Order failed: %s", exc)
                telegram_alert(f"Order failed: {exc}")
                return {"status":"error","error": str(exc)}

# ---------- SUPERVISOR / BOT LOOP ----------
class TradingBot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.exchange = None
        self.model = ModelWrapper()
        self.risk_mgr = RiskManager(cfg)
        self.executor = None
        self.running = False
        init_db()

    def init_exchange(self):
        ex_name = self.cfg["exchange"]
        # ccxt exchange init -- API keys set in env or passed via config securely
        ex_class = getattr(ccxt, ex_name)
        ex = ex_class({
            "enableRateLimit": True,
            # For testnets, client may need urls override - leave for advanced config
        })
        if self.cfg["mode"] in ("paper","backtest"):
            # no keys required for paper mode in this simple prototype
            pass
        else:
            # live mode requires API keys in environment or config (not stored here)
            ex.apiKey = self.cfg.get("apiKey")
            ex.secret = self.cfg.get("secret")
        self.exchange = ex
        self.executor = Executor(self.exchange, self.cfg["symbol"], self.risk_mgr, paper_balance=self.cfg.get("paper_balance", 10000.0))

    def backtest(self):
        logger.info("Starting backtest...")
        self.init_exchange()
        ohlcv = self.exchange.fetch_ohlcv(self.cfg["symbol"], timeframe=self.cfg["timeframe"], limit=1000)
        df = ohlcv_to_df(ohlcv)
        df = add_technical_features(df)
        # Train model
        self.model.train(df)
        # Simulate simple naive backtest
        equity = self.cfg.get("paper_balance", 10000.0)
        records = []
        for idx in range(len(df)-1):
            row = df.iloc[idx]
            pred = self.model.predict(row)
            price = row["close"]
            if pred["signal"] == "buy":
                size = self.risk_mgr.compute_size(equity, price)
                if size <= 0: continue
                # Simulate next-bar result
                next_price = df.iloc[idx+1]["close"]
                pnl = (next_price - price) * size
                equity += pnl
                records.append(pnl)
        logger.info("Backtest finished. Final equity: %.2f, trades: %d", equity, len(records))
        telegram_alert(f"Backtest finished. Final equity: {equity:.2f}, trades: {len(records)}")
        return {"final_equity": equity, "trades": len(records)}

    def run_once(self):
        # single cycle: fetch data, predict, execute if signal + risk checks
        ohlcv = fetch_ohlcv(self.exchange, self.cfg["symbol"], self.cfg["timeframe"], limit=200)
        df = ohlcv_to_df(ohlcv)
        df = add_technical_features(df)
        last_row = df.iloc[-1]
        # if model not trained, train on history
        if not self.model.clf:
            logger.info("Training initial model")
            self.model.train(df)
        pred = self.model.predict(last_row)
        logger.info("Signal=%s conf=%.3f close=%.2f", pred["signal"], pred["confidence"], last_row["close"])
        if pred["signal"] in ("buy","sell") and pred["confidence"]>0.55:
            equity = self.executor.get_equity()
            size = self.risk_mgr.compute_size(equity, last_row["close"])
            if size <= 0:
                logger.warning("Size computed 0, skipping order")
                return
            # safety checks
            if equity <= 0:
                telegram_alert("Equity depleted, stop trading")
                self.running = False
                return
            # place order
            res = self.executor.place_order(pred["signal"], size, price=last_row["close"])
            telegram_alert(f"Order placed ({pred['signal']}) size={size} price={last_row['close']} res={res}")
        else:
            logger.debug("No actionable signal")

    def start_loop(self, interval_seconds=60):
        logger.info("Bot starting main loop in mode=%s", self.cfg["mode"])
        self.init_exchange()
        self.running = True
        while self.running:
            try:
                self.run_once()
            except Exception as e:
                logger.exception("Error in loop: %s", e)
                telegram_alert(f"Bot error: {e}")
                # on repeated errors, stop trading (could be more sophisticated)
                # self.running = False
            time.sleep(interval_seconds)

# ---------- MAIN ----------
def main():
    bot = TradingBot(CONFIG)
    if CONFIG["mode"] == "backtest":
        res = bot.backtest()
        print(res)
        return
    # For paper/live, run loop in background thread then join; here simple run_once every timeframe interval
    bot_thread = threading.Thread(target=bot.start_loop, kwargs={"interval_seconds": 60})
    bot_thread.daemon = True
    bot_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
        bot.running = False
        bot_thread.join()

if __name__ == "__main__":
    main()

