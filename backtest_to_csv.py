import json
import csv

# Load the backtest results
with open("cascade_backtest_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

data = results.get("BTC", {})

# Create the CSV file
with open("backtest_results.csv", "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Header 1: Individual Signals
    writer.writerow(["Individual Signals"])
    writer.writerow(["Signal", "Hit Rate", "Trades"])
    for signal in ["5min_n1", "5min_n2", "1min_n5"]:
        writer.writerow([
            signal.replace("_", ":"),
            data["individual_signals"][signal]["hit_rate"],
            data["individual_signals"][signal]["trades"]
        ])
    writer.writerow([])  # Empty row
    
    # Header 2: Cascade Combined
    writer.writerow(["Cascade Combined"])
    writer.writerow(["Conviction", "Hit Rate", "Trades", "PnL"])
    for conviction in ["HIGH_conviction_3of3", "MEDIUM_conviction_2of3"]:
        writer.writerow([
            conviction,
            data["cascade_combined"][conviction]["hit_rate"],
            data["cascade_combined"][conviction]["trades"],
            f"${data['cascade_combined'][conviction]['pnl']:.2f}"
        ])
    writer.writerow([])  # Empty row
    
    # Header 3: Overall Comparison
    writer.writerow(["Overall Schema Comparison"])
    writer.writerow(["Schema", "Final Bankroll", "Total Return", "Edge Slots"])
    schemas = [
        ("CASCADE HIGH (3/3)", "final_bankroll", "total_return", 21),
        ("5min:n+1 ONLY", "5m_n1_final_bankroll", "5m_n1_total_return", 16),
        ("1min:n+5 ONLY", "1m_n5_final_bankroll", "1m_n5_total_return", 14)
    ]
    for schema_name, bankroll_key, return_key, edge_slots in schemas:
        writer.writerow([
            schema_name,
            f"${data[bankroll_key]:.2f}",
            data[return_key],
            edge_slots
        ])
    writer.writerow([])  # Empty row
    
    # CASCADE HIGH Day-Hour Breakdown
    writer.writerow(["CASCADE HIGH (3/3) Day-Hour Breakdown (Top 10)"])
    writer.writerow(["Day_Hour", "Hit Rate", "Trades", "PnL", "Edge?"])
    dh_breakdown = data["day_hourly_breakdown"]
    dh_sorted = sorted(dh_breakdown.items(), key=lambda x: x[1]["pnl"], reverse=True)
    for day_hour, info in dh_sorted[:10]:
        writer.writerow([
            day_hour,
            info["hit_rate"],
            info["trades"],
            f"${info['pnl']:.2f}",
            "✅" if info.get("edge") else ""
        ])
    writer.writerow([])  # Empty row
    
    # 5min:n+1 Day-Hour Breakdown
    writer.writerow(["5min:n+1 ONLY Day-Hour Breakdown (Top 10)"])
    writer.writerow(["Day_Hour", "Hit Rate", "Trades", "PnL", "Edge?"])
    dh_breakdown_5m = data["5m_n1_day_hourly_breakdown"]
    dh_sorted_5m = sorted(dh_breakdown_5m.items(), key=lambda x: x[1]["pnl"], reverse=True)
    for day_hour, info in dh_sorted_5m[:10]:
        writer.writerow([
            day_hour,
            info["hit_rate"],
            info["trades"],
            f"${info['pnl']:.2f}",
            "✅" if info.get("edge") else ""
        ])
    writer.writerow([])  # Empty row
    
    # 1min:n+5 Day-Hour Breakdown
    writer.writerow(["1min:n+5 ONLY Day-Hour Breakdown (Top 10)"])
    writer.writerow(["Day_Hour", "Hit Rate", "Trades", "PnL", "Edge?"])
    dh_breakdown_1m = data["1m_n5_day_hourly_breakdown"]
    dh_sorted_1m = sorted(dh_breakdown_1m.items(), key=lambda x: x[1]["pnl"], reverse=True)
    for day_hour, info in dh_sorted_1m[:10]:
        writer.writerow([
            day_hour,
            info["hit_rate"],
            info["trades"],
            f"${info['pnl']:.2f}",
            "✅" if info.get("edge") else ""
        ])

print("CSV file created: backtest_results.csv")
print("Now you can upload this CSV to Google Sheets!")
