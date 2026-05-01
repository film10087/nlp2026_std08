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

test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้าโดยไม่ได้รับอนุญาตตามมาตรา 27 แห่ง พ.ร.บ. สิทธิบัตร"
tokens = legal_tokenizer(test_text)
print(f"Input: {test_text}")
print(f"Output: {tokens}")

# การสวัดค่า ความกำกวม (Ambiguity Rate) เทียบระหว่าง Dictionary Base+Regex กับ "wangchanberta"
def calculate_baseline_ambiguity_(text):
    matches = []
    for word in LEGAL_KEYWORDS:
        for m in re.finditer(re.escape(word), text):
            if m:
                matches.append((m.start(), m.end(), word))

            #ตรวจสอบการซ้อน  (Overlapping) ของ matches
    overlaps = 0
    for i in range(len(matches)):
        for j in range(i+1, len(matches)):
            # ถ้าตำแหน่งเริ่ม/จบ ทับซ้อนกัน ถือว่ากำกวม
            if matches[i][0] < matches[j][1] and matches[j][0] < matches[i][1]:
                overlaps += 1
    return overlaps / len(matches) if matches else 0
#รัน แสดงผล Baseline
sample_text = "คดีการละเมิดสิทธิบัตรและเครื่องหมายการค้า"
baseline_tokens = legal_tokenizer(sample_text)
baseline_rate = calculate_baseline_ambiguity_(sample_text)
print(f"W1 Baseline Result:")
print(f"W1 Baseline Tokens: {baseline_tokens}")
print(f"Bseline Ambiguity Rate: {baseline_rate}")

# WangChanBERTa Pretain
from transformers import AutoTokenizer
# 1. load WangchanBerta 
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def berta_tokenizer(text):
    tokens = tokenizer.tokenize(text)
    return [t.replace("▁","") for t in tokens if t.replace("▁","")]
# ทดสอบการวัด กำกวม เน้นด้านกฎหมาย
def analyze_refined_ambiguity(text,legal_keywords):
    tokens = berta_tokenizer(text)
    frag_score = []
    for kw in legal_keywords:
        if kw in text:
            kw_tokens = berta_tokenizer(kw)
            fragment_ratio = len(kw_tokens) /1
            frag_score.append(fragment_ratio)
    # Ambiguuity = Average Fragmentation - 1 แต่ถ้าตัดพอดี เท่ากับ 0
    avg_frag = (sum(frag_score)/len(frag_score)) -1 if frag_score else 0
    return min (avg_frag,1.0) 
# รันแสดงผลเปรียบเทียบ
refined_tokens = berta_tokenizer(sample_text)
refined_rate = analyze_refined_ambiguity(sample_text,LEGAL_KEYWORDS)
print(f"--W1 : Refined With WangChanBERTa ---")
print(f"Tokens: {refined_tokens}")
print(f"new Ambiguity fragmentation Rate: {refined_rate:.3f}")



# 2. Context-aware Entity Extraction
def extract_legal_entities(text):
    entities = []
    # จำลองหาความผิด ประเภทของ IP (IP Type) และหาการกระทำ (Action)
    if "สิทธิบัตร" in text:
        entities.append({"type":"IP_Type", "value": "PATENT", "conf":0.95})
    if "ละเมิด" in text:
        entities.append({"type":"Action", "value": "VIOLATION", "conf":0.85})
    return entities

sample = "มีการละเมิดสิทธิบัตรรายใหญ่เกิดขึ้น"
found = extract_legal_entities(sample)
print(f"---Entity Extraction---")
for e in found:
    print(f"{e["type"]} {e["value"]} Confidence: {e['conf']}")

# 3. Feature Engineering (TF-IDF Base)
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    "ละเมิดสิทธิบัตรเเครื่องหมายการค้า",
    "การกระทำความผิดฐานละเมิดสิทธิบัตร",
    "จำเลยถูกฟ้องละเมิดสิทธิบัตร"
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
