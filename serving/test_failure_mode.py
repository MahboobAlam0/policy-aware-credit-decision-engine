from serving.api_contract import CreditDecisionResponse
from pydantic import ValidationError

def test_extreme_pd_failure():
    print("--- RUNNING INTENTIONAL FAILURE MODE TEST ---")
    print("Scenario: Model outputs an uncalibrated PD score > 1.0 (Extreme Input Mismatch)")
    
    try:
        response = CreditDecisionResponse(
            pd_score=1.15,  # INTENTIONAL FAILURE (Must be <= 1.0)
            decision_tier="APPROVE",
            reason_codes=["Income Stability (DECREASED RISK)"]
        )
        print("FAIL: The system swallowed the bad input silently!")
    except ValidationError as e:
        print("\nSUCCESS! The system failed loudly and predictably. Validation Error caught:")
        print(f"Error Details:\n{e}")
        
    print("\nWhy this matters:")
    print("Silent failures in credit decision pipelines can lead to catastrophic underwriting ")
    print("of uncalibrated scores. Strict Pydantic contracts ensure that artifact drift ")
    print("or extreme data anomalies instantly trip alarms rather than originating bad loans.")

if __name__ == "__main__":
    test_extreme_pd_failure()
