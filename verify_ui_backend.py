import requests
import json

url = "http://localhost:8000/predict"
payload = {
    "credit_score": 650,
    "age": 55,  # Should trigger "Senior Customer"
    "tenure": 5,
    "balance": 0, # Should trigger "Zero Balance"
    "products_number": 1,
    "credit_card": 1,
    "active_member": 0, # Should trigger "Inactive Member"
    "estimated_salary": 50000
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    
    print("\n--- API Response ---")
    print(json.dumps(data, indent=2))
    
    # Assertions
    assert "factors" in data, "Response missing 'factors' field"
    assert "Senior Customer (>50)" in data["factors"], "Missing Senior factor"
    assert "Inactive Member" in data["factors"], "Missing Inactive factor"
    
    print("\n✅ UI Backend Verification Passed!")
    
except Exception as e:
    print(f"\n❌ Verification Failed: {e}")
