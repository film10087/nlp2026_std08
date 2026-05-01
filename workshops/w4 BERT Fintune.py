import numpy as np

# --- 1. Mock Tokenizer (สำหรับจำลองการแปลงข้อความ) ---
class MockTokenizer:
    def encode(self, texts, max_length=32):
        batch_size = len(texts)
        # จำลองการสร้าง input_ids และ attention_mask
        input_ids = np.random.randint(100, 10000, size=(batch_size, max_length))
        attention_mask = np.ones((batch_size, max_length))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

# --- 2. Mock BERT Encoder (จำลอง Transformer 12 Layers) ---
class MockBERTEncoder:
    """
    จำลอง BERT 12-layer encoder
    Output: h_cls shape (batch, 768)
    """
    def __init__(self, hidden_size=768, seed=42):
        self.hidden_size = hidden_size
        self.rng = np.random.RandomState(seed)

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        # จำลองค่า h_cls (Hidden state ของ [CLS] token)
        h_cls = self.rng.randn(batch, self.hidden_size).astype(np.float32) * 0.1
        return h_cls

# --- 3. Classification Head (The Judge) ---
class ClassificationHead:
    """
    Linear layer สำหรับจำแนก class (PATENT, COPYRIGHT, NONE)
    """
    def __init__(self, hidden_size=768, n_classes=3, dropout=0.1, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier Initialization สำหรับน้ำหนัก
        s = np.sqrt(2.0 / (hidden_size + n_classes))
        self.W = rng.randn(n_classes, hidden_size).astype(np.float32) * s
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.dropout = dropout

    def forward(self, h_cls, training=False):
        # Dropout: ใช้เฉพาะช่วง training เพื่อป้องกัน Overfitting
        if training:
            mask = (np.random.rand(*h_cls.shape) > self.dropout).astype(np.float32)
            h_cls = h_cls * mask / (1 - self.dropout)

        # คำนวณ Logits: (batch, 768) @ (768, 3) = (batch, 3)
        logits = h_cls @ self.W.T + self.b
        
        # Softmax: แปลงเป็นความน่าจะเป็น
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

# --- 4. BERT For Classification (Main Class) ---
class BERTForClassification:
    CLASS_NAMES = ["ละเมิด_สิทธิบัตร", "ละเมิด_ลิขสิทธิ์", "ไม่ละเมิด"]

    def __init__(self, seed=42):
        self.encoder = MockBERTEncoder(seed=seed)
        self.head = ClassificationHead(seed=seed)
        self.tokenizer = MockTokenizer()

    def predict_proba(self, input_ids, attention_mask):
        h_cls = self.encoder.forward(input_ids, attention_mask)
        return self.head.forward(h_cls)

    def predict(self, input_ids, attention_mask):
        return np.argmax(self.predict_proba(input_ids, attention_mask), axis=1)

    def show_prediction(self, texts):
        # ขั้นตอน Tokenization
        enc = self.tokenizer.encode(texts, max_length=32)
        
        # ขั้นตอนการทำนาย
        prob = self.predict_proba(enc["input_ids"], enc["attention_mask"])
        pred = np.argmax(prob, axis=1)

        # แสดงผลลัพธ์
        print(f"\n{'ข้อความ':<45} {'การทำนาย':<20} {'ความมั่นใจ'}")
        print(f"{'─'*75}")
        for text, p, pr in zip(texts, pred, prob):
            confidence = pr[p] * 100
            print(f"{text[:43]:<45} {self.CLASS_NAMES[p]:<20} {confidence:.1f}%")

# --- 5. Execution Block ---
if __name__ == "__main__":
    # เริ่มต้นโมเดล
    model = BERTForClassification(seed=42)
    
    # ตัวอย่างข้อมูลทดสอบ (Legal Context)
    test_cases = [
        "ผู้ต้องหานำเข้าสินค้าปลอมแปลงสิทธิบัตร",
        "จำเลยทำซ้ำงานที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาต",
        "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรถูกต้องแล้ว",
        "มีการดัดแปลงโปรแกรมคอมพิวเตอร์เพื่อการค้า"
    ]
    
    model.show_prediction(test_cases)