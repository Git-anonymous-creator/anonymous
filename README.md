# Anonymous: CognitiveContinuum

- This project provides an interactive **fuzzy logic-based intent management system**
- It integrates decision-making, system monitoring, and control logic using **Python**, **skfuzzy**, and **Tkinter**.
- The config.py file includes servers IP addresses and values 
- The main.py includes the main code of CognitiveContinuum
- The fixed-files folder includes the neccessary files that are read when running the code
- Note- You may need the API Key for GPT-4o to run the code.



<< FUZZY LOGIC RULES AND MEMBERSHIP FUNCTIONS >>
==================

Fuzzy Logic Rules:
__________________
Rule 1: IF response_time=low, cost=low, power=low, reliability=low THEN provisioning_strategy=few, placement_strategy=low

Rule 2: IF response_time=low, cost=low, power=low, reliability=medium THEN provisioning_strategy=few, placement_strategy=medium

Rule 3: IF response_time=low, cost=low, power=low, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 4: IF response_time=low, cost=low, power=medium, reliability=low THEN provisioning_strategy=few, placement_strategy=high

Rule 5: IF response_time=low, cost=low, power=medium, reliability=medium THEN provisioning_strategy=few, placement_strategy=high

Rule 6: IF response_time=low, cost=low, power=medium, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 7: IF response_time=low, cost=low, power=high, reliability=low THEN provisioning_strategy=few, placement_strategy=high

Rule 8: IF response_time=low, cost=low, power=high, reliability=medium THEN provisioning_strategy=few, placement_strategy=high

Rule 9: IF response_time=low, cost=low, power=high, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 10: IF response_time=low, cost=medium, power=low, reliability=low THEN provisioning_strategy=few, placement_strategy=medium

Rule 11: IF response_time=low, cost=medium, power=low, reliability=medium THEN provisioning_strategy=few, placement_strategy=medium

Rule 12: IF response_time=low, cost=medium, power=low, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 13: IF response_time=low, cost=medium, power=medium, reliability=low THEN provisioning_strategy=few, placement_strategy=medium

Rule 14: IF response_time=low, cost=medium, power=medium, reliability=medium THEN provisioning_strategy=few, placement_strategy=medium

Rule 15: IF response_time=low, cost=medium, power=medium, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 16: IF response_time=low, cost=medium, power=high, reliability=low THEN provisioning_strategy=few, placement_strategy=low

Rule 17: IF response_time=low, cost=medium, power=high, reliability=medium THEN provisioning_strategy=few, placement_strategy=medium

Rule 18: IF response_time=low, cost=medium, power=high, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 19: IF response_time=low, cost=high, power=low, reliability=low THEN provisioning_strategy=few, placement_strategy=low

Rule 20: IF response_time=low, cost=high, power=low, reliability=medium THEN provisioning_strategy=few, placement_strategy=low

Rule 21: IF response_time=low, cost=high, power=low, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 22: IF response_time=low, cost=high, power=medium, reliability=low THEN provisioning_strategy=few, placement_strategy=low

Rule 23: IF response_time=low, cost=high, power=medium, reliability=medium THEN provisioning_strategy=few, placement_strategy=low

Rule 24: IF response_time=low, cost=high, power=medium, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 25: IF response_time=low, cost=high, power=high, reliability=low THEN provisioning_strategy=few, placement_strategy=low

Rule 26: IF response_time=low, cost=high, power=high, reliability=medium THEN provisioning_strategy=few, placement_strategy=medium

Rule 27: IF response_time=low, cost=high, power=high, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 28: IF response_time=medium, cost=low, power=low, reliability=low THEN provisioning_strategy=medium, placement_strategy=medium

Rule 29: IF response_time=medium, cost=low, power=low, reliability=medium THEN provisioning_strategy=medium, placement_strategy=medium

Rule 30: IF response_time=medium, cost=low, power=low, reliability=high THEN provisioning_strategy=medium, placement_strategy=high

Rule 31: IF response_time=medium, cost=low, power=medium, reliability=low THEN provisioning_strategy=medium, placement_strategy=medium

Rule 32: IF response_time=medium, cost=low, power=medium, reliability=medium THEN provisioning_strategy=moderate, placement_strategy=medium

Rule 33: IF response_time=medium, cost=low, power=medium, reliability=high THEN provisioning_strategy=medium, placement_strategy=high

Rule 34: IF response_time=medium, cost=low, power=high, reliability=low THEN provisioning_strategy=medium, placement_strategy=low

Rule 35: IF response_time=medium, cost=low, power=high, reliability=medium THEN provisioning_strategy=medium, placement_strategy=low

Rule 36: IF response_time=medium, cost=low, power=high, reliability=high THEN provisioning_strategy=moderate, placement_strategy=high

Rule 37: IF response_time=medium, cost=medium, power=low, reliability=low THEN provisioning_strategy=medium, placement_strategy=low

Rule 38: IF response_time=medium, cost=medium, power=low, reliability=medium THEN provisioning_strategy=medium, placement_strategy=medium

Rule 39: IF response_time=medium, cost=medium, power=low, reliability=high THEN provisioning_strategy=medium, placement_strategy=high

Rule 40: IF response_time=medium, cost=medium, power=medium, reliability=low THEN provisioning_strategy=moderate, placement_strategy=low

Rule 41: IF response_time=medium, cost=medium, power=medium, reliability=medium THEN provisioning_strategy=moderate, placement_strategy=medium

Rule 42: IF response_time=medium, cost=medium, power=medium, reliability=high THEN provisioning_strategy=moderate, placement_strategy=high

Rule 43: IF response_time=medium, cost=medium, power=high, reliability=low THEN provisioning_strategy=moderate, placement_strategy=low

Rule 44: IF response_time=medium, cost=medium, power=high, reliability=medium THEN provisioning_strategy=moderate, placement_strategy=low

Rule 45: IF response_time=medium, cost=medium, power=high, reliability=high THEN provisioning_strategy=moderate, placement_strategy=high

Rule 46: IF response_time=medium, cost=high, power=low, reliability=low THEN provisioning_strategy=moderate, placement_strategy=low

Rule 47: IF response_time=medium, cost=high, power=low, reliability=medium THEN provisioning_strategy=moderate, placement_strategy=medium

Rule 48: IF response_time=medium, cost=high, power=low, reliability=high THEN provisioning_strategy=moderate, placement_strategy=high

Rule 49: IF response_time=medium, cost=high, power=medium, reliability=low THEN provisioning_strategy=moderate, placement_strategy=low

Rule 50: IF response_time=medium, cost=high, power=medium, reliability=medium THEN provisioning_strategy=moderate, placement_strategy=low

Rule 51: IF response_time=medium, cost=high, power=medium, reliability=high THEN provisioning_strategy=few, placement_strategy=high

Rule 52: IF response_time=medium, cost=high, power=high, reliability=low THEN provisioning_strategy=moderate, placement_strategy=low

Rule 53: IF response_time=medium, cost=high, power=high, reliability=medium THEN provisioning_strategy=few, placement_strategy=medium

Rule 54: IF response_time=medium, cost=high, power=high, reliability=high THEN provisioning_strategy=moderate, placement_strategy=high

Rule 55: IF response_time=high, cost=low, power=low, reliability=low THEN provisioning_strategy=many, placement_strategy=low

Rule 56: IF response_time=high, cost=low, power=low, reliability=medium THEN provisioning_strategy=many, placement_strategy=medium

Rule 57: IF response_time=high, cost=low, power=low, reliability=high THEN provisioning_strategy=many, placement_strategy=high

Rule 58: IF response_time=high, cost=low, power=medium, reliability=low THEN provisioning_strategy=many, placement_strategy=medium

Rule 59: IF response_time=high, cost=low, power=medium, reliability=medium THEN provisioning_strategy=many, placement_strategy=medium

Rule 60: IF response_time=high, cost=low, power=medium, reliability=high THEN provisioning_strategy=many, placement_strategy=high

Rule 61: IF response_time=high, cost=low, power=high, reliability=low THEN provisioning_strategy=many, placement_strategy=low

Rule 62: IF response_time=high, cost=low, power=high, reliability=medium THEN provisioning_strategy=many, placement_strategy=medium

Rule 63: IF response_time=high, cost=low, power=high, reliability=high THEN provisioning_strategy=many, placement_strategy=high

Rule 64: IF response_time=high, cost=medium, power=low, reliability=low THEN provisioning_strategy=many, placement_strategy=medium

Rule 65: IF response_time=high, cost=medium, power=low, reliability=medium THEN provisioning_strategy=many, placement_strategy=medium

Rule 66: IF response_time=high, cost=medium, power=low, reliability=high THEN provisioning_strategy=many, placement_strategy=high

Rule 67: IF response_time=high, cost=medium, power=medium, reliability=low THEN provisioning_strategy=many, placement_strategy=medium

Rule 68: IF response_time=high, cost=medium, power=medium, reliability=medium THEN provisioning_strategy=many, placement_strategy=medium

Rule 69: IF response_time=high, cost=medium, power=medium, reliability=high THEN provisioning_strategy=many, placement_strategy=high

Rule 70: IF response_time=high, cost=medium, power=high, reliability=low THEN provisioning_strategy=many, placement_strategy=low

Rule 71: IF response_time=high, cost=medium, power=high, reliability=medium THEN provisioning_strategy=many, placement_strategy=medium

Rule 72: IF response_time=high, cost=medium, power=high, reliability=high THEN provisioning_strategy=many, placement_strategy=high

Rule 73: IF response_time=high, cost=high, power=low, reliability=low THEN provisioning_strategy=many, placement_strategy=low

Rule 74: IF response_time=high, cost=high, power=low, reliability=medium THEN provisioning_strategy=many, placement_strategy=low

Rule 75: IF response_time=high, cost=high, power=low, reliability=high THEN provisioning_strategy=many, placement_strategy=high

Rule 76: IF response_time=high, cost=high, power=medium, reliability=low THEN provisioning_strategy=many, placement_strategy=low

Rule 77: IF response_time=high, cost=high, power=medium, reliability=medium THEN provisioning_strategy=many, placement_strategy=medium

Rule 78: IF response_time=high, cost=high, power=medium, reliability=high THEN provisioning_strategy=many, placement_strategy=high

Rule 79: IF response_time=high, cost=high, power=high, reliability=low THEN provisioning_strategy=many, placement_strategy=low

Rule 80: IF response_time=high, cost=high, power=high, reliability=medium THEN provisioning_strategy=many, placement_strategy=low

Rule 81: IF response_time=high, cost=high, power=high, reliability=high THEN provisioning_strategy=many, placement_strategy=high



Membership Functions:
_____________________
membership_functions = {
    "response_time": {"low": [0, 0, 0.2, 0.4], "medium": [0.2, 0.4, 0.6, 0.8], "high": [0.6, 0.8, 1, 1]},
    
    "cost": {"low": [0, 0, 0.2, 0.4], "medium": [0.2, 0.4, 0.6, 0.8], "high": [0.6, 0.8, 1, 1]},
    
    "power": {"low": [0, 0, 0.2, 0.4], "medium": [0.2, 0.4, 0.6, 0.8], "high": [0.6, 0.8, 1, 1]},
    
    "reliability": {"low": [0, 0, 0.2, 0.4], "medium": [0.2, 0.4, 0.6, 0.8], "high": [0.6, 0.8, 1, 1]},
    
    "provisioning_strategy": {"few": [0, 0, 0.2, 0.4], "moderate": [0.2, 0.4, 0.6, 0.8], "many": [0.6, 0.8, 1, 1]},
    
    "placement_strategy": {"low": [0, 0, 0.2, 0.4], "medium": [0.2, 0.4, 0.6, 0.8], "high": [0.6, 0.8, 1, 1]}
    
}

