# Demo model used when no sklearn model is provided.
# This avoids depending on scikit-learn at runtime and provides deterministic predictions
# so the hosted app can demonstrate model.predict behavior without external model artifacts.

class DemoEncoder:
    def __init__(self, classes=None):
        self.classes_ = classes or [
            'IndiGo', 'Air India', 'SpiceJet', 'Vistara', 'GoAir', 'AirAsia India', 'Alliance Air', 'TruJet'
        ]


class DemoModel:
    def __init__(self, airlines=None):
        self.airlines = airlines or [
            'IndiGo', 'Air India', 'SpiceJet', 'Vistara', 'GoAir', 'AirAsia India', 'Alliance Air', 'TruJet'
        ]

    def predict(self, X):
        # X is expected to be a pandas DataFrame with columns like 'Airline', 'Duration', 'Stops'
        prices = []
        for _, row in X.iterrows():
            base = 2500
            duration = float(row.get('Duration', 60))
            stops = int(row.get('Stops', 0))
            airline = row.get('Airline', '')
            airline_modifier = (self.airlines.index(airline) + 1) * 150 if airline in self.airlines else (len(airline) % 7) * 100
            price = base + duration * 8 + stops * 600 + airline_modifier
            # small deterministic adjustment
            price *= 1 + ((self.airlines.index(airline) - 2) * 0.02) if airline in self.airlines else price
            prices.append(round(price, 0))
        return prices
