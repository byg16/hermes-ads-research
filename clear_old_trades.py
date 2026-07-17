"""Delete old trade data from cascade_trades.db"""
import sqlite3

DB_PATH = "cascade_trades.db"

def clear_trades():
    conn = sqlite3.connect(DB_PATH)
    try:
        # First create table if it doesn't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cascade_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                asset TEXT,
                p_up_5m_n1 REAL,
                p_up_5m_n2 REAL,
                p_up_1m_n5 REAL,
                combined_p_up REAL,
                conviction TEXT,
                side TEXT,
                kelly_fraction REAL,
                stake_usd REAL,
                correct INTEGER,
                pnl_usd REAL,
                bankroll REAL,
                hour_utc INTEGER,
                rationale TEXT
            )
        """)
        # Delete all records from cascade_trades table
        conn.execute("DELETE FROM cascade_trades")
        # Reset auto-increment counter
        conn.execute("DELETE FROM sqlite_sequence WHERE name='cascade_trades'")
        conn.commit()
        print("[SUCCESS] Successfully deleted all old trade data!")
    except Exception as e:
        print(f"[ERROR] Error deleting old trade data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    # Bypass confirmation for scripted use
    clear_trades()
