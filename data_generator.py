
import pandas as pd
import numpy as np
import random

def generate_synthetic_data(num_samples=5000):
    """
    Generates a synthetic dataset for Airline Food Demand Prediction
    satisfying all constraints specified in the requirements.
    """
    
    # Initialize lists to store data
    data = []
    
    for i in range(num_samples):
        flight_id = 1000 + i
        
        # Determine if international (ensure at least 15% are international)
        # Using 30% to be safe and ensure diversity
        is_international = 1 if random.random() < 0.3 else 0
        
        # Flight Duration
        # Constraint: if is_international == 1 then flight_duration >= 3
        if is_international:
            flight_duration = round(random.uniform(3.0, 12.0), 1)
        else:
            # Domestic/Short haul: 1-12 is allowed, but typically shorter
            # Requirement: "flight_duration must include both short (1-3h) and long (8-12h) flights"
            # Randomly picking from full range 1-12 to ensure coverage
            flight_duration = round(random.uniform(1.0, 12.0), 1)
            
        # Passenger Count (50 - 300)
        passenger_count = random.randint(50, 300)
        
        # Adult vs Child Ratio
        # Random ratio for children, typically low (0-20%)
        child_ratio = random.uniform(0, 0.2)
        child_passengers = int(passenger_count * child_ratio)
        adult_passengers = passenger_count - child_passengers
        
        # Business Class Ratio (0 - 1)
        # Typically 0-30%
        business_class_ratio = round(random.uniform(0, 0.3), 2)
        
        # --- Target Variable Calculation ---
        # Critical Requirement: Depends on at least 3 features.
        # Logic:
        # Base demand: Every passenger needs at least a snack/meal?
        # Duration factor: Longer flights = more food needed per person (e.g., 2 meals for long haul)
        # Class factor: Business class might have more elaborate/multiple courses or guaranteed availability
        # Type factor: International flights often have more standardized full meals
        
        # Base demand calculation
        base_demand = passenger_count 
        
        # Logic 1: Duration Multiplier
        # Short (<3h): 0.6 units/pax (maybe just snacks + drink)
        # Medium (3-7h): 1.0 units/pax (1 meal)
        # Long (>7h): 1.5 units/pax (1 meal + snack/breakfast)
        if flight_duration < 3:
            duration_factor = 0.6
        elif flight_duration < 7:
            duration_factor = 1.0
        else:
            duration_factor = 1.5
            
        # Logic 2: Business Class Adder
        # Business class passengers consume ~1.2x - 1.5x regular 'units' (more courses, wastage assumption)
        business_weight = 1 + business_class_ratio * 0.5 
        
        # Logic 3: International Adder
        # International flights often load extra buffer
        inter_factor = 1.1 if is_international else 1.0
        
        # Combine
        # randomness to simulate real world uncertainty (+/- 10%)
        noise = random.uniform(0.9, 1.1)
        
        estimated_demand = (base_demand * duration_factor * business_weight * inter_factor) * noise
        
        # Ensure integer
        total_food_demand = int(np.ceil(estimated_demand))
        
        # Constraint: total_food_demand >= passenger_count * 0.5
        if total_food_demand < passenger_count * 0.5:
            total_food_demand = int(np.ceil(passenger_count * 0.5))
            
        # Append record
        row = {
            'flight_id': flight_id,
            'flight_duration': flight_duration,
            'passenger_count': passenger_count,
            'adult_passengers': adult_passengers,
            'child_passengers': child_passengers,
            'business_class_ratio': business_class_ratio,
            'is_international': is_international,
            'total_food_demand': total_food_demand
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Final Validation Checks
    print("--- Validation Checks ---")
    print(f"Row count >= 5000: {len(df) >= 5000} ({len(df)})")
    
    valid_passengers = (df['adult_passengers'] + df['child_passengers'] == df['passenger_count']).all()
    print(f"adult + child == passenger_count: {valid_passengers}")
    
    valid_duration = ((df['flight_duration'] >= 1) & (df['flight_duration'] <= 12)).all()
    print(f"1 <= duration <= 12: {valid_duration}")
    
    # Check Constraint 4: if is_international == 1 then flight_duration >= 3
    invalid_international = df[(df['is_international'] == 1) & (df['flight_duration'] < 3)]
    print(f"International flights >= 3h: {len(invalid_international) == 0}")
    
    valid_food = (df['total_food_demand'] >= df['passenger_count'] * 0.5).all()
    print(f"Food demand >= 0.5 * passengers: {valid_food}")
    
    inter_ratio = df['is_international'].mean()
    print(f"International Ratio >= 15%: {inter_ratio >= 0.15} ({inter_ratio:.2%})")
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(5500) # Generate 5500 to be safe
    df.to_csv("synthetic_flight_data.csv", index=False)
    print("\nDataset saved to 'synthetic_flight_data.csv'")
