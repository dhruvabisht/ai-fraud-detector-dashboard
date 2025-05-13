import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_data(n=5000):
    data = []
    start_date = datetime.now() - timedelta(days=30)
    for _ in range(n):
        transaction = {
            "Transaction_ID": fake.uuid4(),
            "User_ID": fake.uuid4(),
            "Merchant_ID": fake.uuid4(),
            "Amount": round(random.uniform(5, 1000), 2),
            "Timestamp": (start_date + timedelta(seconds=random.randint(0, 2592000))).isoformat(),
            "Latitude": round(random.uniform(52.0, 53.5), 6),
            "Longitude": round(random.uniform(-8.0, -6.0), 6),
            "Device_Type": random.choice(["Mobile", "Desktop", "POS"]),
            "Transaction_Type": random.choice(["Online", "In-store"]),
        }
        data.append(transaction)
    df = pd.DataFrame(data)
    df.to_csv("/Applications/Dhruva/Fraud detector using AI/data/synthetic_transactions.csv", index=False)

if __name__ == "__main__":
    generate_data()
