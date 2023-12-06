from datetime import datetime, timedelta

def generate_data(start_date, end_date, interval_minutes):
    current_date = start_date
    slope = 500 / ((end_date - start_date).days * 24 * 60 / interval_minutes)
    data = []

    while current_date <= end_date:
        if 4 <= current_date.hour < 20:  # Ensure time is between 4 am and 8 pm
            value = 30 if current_date <= start_date + timedelta(days=2) else round(slope * (current_date - (start_date + timedelta(days=2))).total_seconds() / (60*interval_minutes)) + 30
            row = f"{current_date.strftime('%Y-%m-%d %H:%M:%S')},{value:.4f},{value:.4f},{value:.4f},{value:.4f},220"
            data.append(row)

        current_date += timedelta(minutes=interval_minutes)

    return data

def save_to_txt(data, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(data))

if __name__ == "__main__":
    start_date = datetime(2021, 3, 16, 4, 0, 0)
    end_date = start_date + timedelta(days=7)  # 7 days total
    interval_minutes = 6  # 1 minute intervals

    generated_data = generate_data(start_date, end_date, interval_minutes)
    save_to_txt(generated_data, "../../data/input/us3000_tickers_A-B_1min_iqksn/art2_1min.txt")