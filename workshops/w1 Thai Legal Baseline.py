# การตัดคำ Tokenization + custom_Dict
import re
from pythainlp.tokenize import word_tokenize

LEGAL_KEYWORDS = ["ละเมิดสิทธิบัตร","เครื่องหมายการค้า","ลิขสิทธิ์","การกระทำความผิด","จำเลย","ศาล","คำพิพากษา","มาตรา","พ.ร.บ."]

def legal_tokenizer(text):
    # 1.Protect Compound Keywords ด้วย Placeholder
    sorted_kw = sorted(LEGAL_KEYWORDS,key=len,reverse=True)
    placeholders = {}
    protected = text
    for i, kw in enumerate(sorted_kw):
        ph = f"__KW{i}__"
        if kw in protected:
            placeholders[ph] = kw
            protected = protected.replace(kw,ph)
    # 2. tokenize ด้วย pythainlp
    tokens_raw = word_tokenize(protected,engine="newmm",keep_whitespace=False)
    
    # 3. restore placeholder
    return [placeholders.get(t,t) for t in tokens_raw]

test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)
print(f"Input: {test_text}")
print(f"Output: {tokens}")

# 2. Context-aware Entity Extraction
def extract_legal_entities(text):
    entities = []
    # จำลองหาความผิด ประเภทของ IP (IP Type) และหาการกระทำ (Action)
    if "สิทธิบัตร" in text:
        entities.append({"type":"IP_Type", "value": "PATENT", "conf":0.95})
    if "ละเมิด" in text:
        entities.append({"type":"Action", "value": "VIOLATION", "conf":0.85})
    return entities

sample = "มีการละเมิดสิทธิบัตรเกิดขึ้นในเขตพื้นที่"
found = extract_legal_entities(sample)
print(f"---Entity Extraction---")
for e in found:
    print(f"{e["type"]} {e["value"]} Confidence: {e['conf']}")

# 3. Feature Engineering (TF-IDF Base)
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    "ละเมิดสิทธิบัตรเเครื่องหมายการค้า",
    "การกระทำความผิด ลิขสิทธิ์",
    "จำเลย ละเมิด ลิขสิทธิ์"
]
# สร้าง vectorizer โดยใช้ Tokenizer ที่สร้างเอง
vectorizer = TfidfVectorizer(tokenizer=legal_tokenizer, token_pattern=None)
tfudf_matrix = vectorizer.fit_transform(corpus)

print(f"---TF-IDF Vector (shape: {tfudf_matrix.shape}) ---")
print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
print(f"Vector sample (Doc 1):\n{tfudf_matrix[1].toarray()}")

# 4. Physics Gate Weight (Legal Hierarchy)
def compute_physics_gate_weight(entities):
    base_weight = 5.0
    for e in entities:
         if e["value"] == "PATENT": base_weight += 2.0 #สิทธิบัตรน้ำหนักสูง
         if e["value"] == "INFRINGEMENT": base_weight += 1.5 
    return min(base_weight,10.0) #maximum  = 10
# ทดสอบคำนวณค่าน้ำหนัก entities แล้วสกัดได้
weight = compute_physics_gate_weight(found)
print(f"--Physics Gate Bridge --")
print(f"legal context weight: {weight:.2f}/10") 
print(f"status: {'High Alert - Trigger Sensor' if weight >=7 else 'Normal Monitoring'}")   



# ตัดด้วย Deep Learning (Deepcut)
# import deepcut
# print(f"Output Deepcut: {deepcut.tokenize(test_text)}")
