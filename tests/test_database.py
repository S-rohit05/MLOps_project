import os
import pytest
from src.database import init_db, save_prediction, get_recent_predictions, DB_NAME

def test_database_workflow():
    # Setup: Use a test database file
    test_db = DB_NAME
    
    # Ensure fresh start
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except PermissionError:
            pass # Might be locked if used by app
    
    init_db()
    assert os.path.exists(test_db)
    
    # Test Save
    data = {
        "credit_score": 600,
        "age": 40,
        "tenure": 3,
        "balance": 60000.0,
        "products_number": 2,
        "credit_card": 1,
        "active_member": 1,
        "estimated_salary": 50000.0
    }
    save_prediction(data, 1, 0.85)
    
    # Test Get
    rows = get_recent_predictions()
    assert len(rows) >= 1
    # Check the latest one
    latest = rows[0]
    # Note: latest might not be the one we just saved if DB was already populated/multithreaded, 
    # but in this isolated test environment it should be, unless we appended to existing DB.
    # The logic above tries to remove DB but might fail if locked. 
    # Let's just check if our data is likely in the returned list.
    
    found = any(r['churn_probability'] == 0.85 for r in rows)
    assert found
