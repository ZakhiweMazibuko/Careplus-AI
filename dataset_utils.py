import pandas as pd

def str_to_bool(val):
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}

def normalize_text(s):
    if pd.isna(s):
        return ""
    return " ".join(str(s).lower().strip().split())

def load_dataset(path="Diseases_Symptoms.csv"):
    data = pd.read_csv(path).fillna("")
    
    # Standardize booleans
    data["Contagious"] = data["Contagious"].apply(str_to_bool)
    data["Chronic"] = data["Chronic"].apply(str_to_bool)
    
    # Normalize text fields
    data["Symptoms"] = data["Symptoms"].apply(normalize_text)
    data["Treatments"] = data["Treatments"].astype(str).apply(lambda s: s.strip())
    data["Name"] = data["Name"].astype(str).apply(lambda s: s.strip())
    data["Disease_Code"] = data["Disease_Code"].astype(str).apply(lambda s: s.strip())
    
    return data
