def categorize(text):
    categories = {
        "malware": "Malware",
        "phishing": "Phishing",
        "ransomware": "Ransomware",
        "trojan": "Trojan",
        "worm": "Worm",
        "spyware": "Spyware",
        "ddos": "DDoS",
        "distributed denial of service": "DDoS",
        "zero day": "Zero Days",
        "data breach": "Data Breach",
        "social engineering": "Social Engineering"
    }
    lower_text = text.lower()
    for keyword, category in categories.items():
        if keyword in lower_text:
            return category
    return "Other" 

def labeler(df):
    df["label"] = df["text"].apply(categorize)
    return df