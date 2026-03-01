# Agentic Log-based Anomaly Detection — 描述式大綱（Coding Blueprint）

> 你可以直接照著本大綱開始寫程式：每一節都包含 **要做什麼、輸入/輸出、MVP、完成判準**。  
> 建議先用 **BGL** 跑通，再擴展到 **HDFS / Thunderbird**。

---

## 0. 專案目標與輸出物

**目標**：在 AllLinLog 高偵測效能的前提下，對 *predicted anomaly sessions* 產出 **evidence-grounded、可追溯（traceable）** 的解釋。  

**兩個系統版本**：
- **Baseline**：Explain-All（所有 predicted anomalies 都解釋）
- **進階**：Budgeted Explain（margin gating：只解釋低信心 anomalies）

**最終輸出**（每個 session 一筆結果）：
- `session_id, pred, logits/prob, (optional: margin), retrieved_evidence_ids, explanation_json, costs(tokens/latency)`

---

## 1. 資料介面層：Session Dataset Loader

### 1.1 要做什麼
把 BGL/HDFS 的資料轉成一致的 session 物件，供後面所有模組使用。

### 1.2 介面定義（建議）
每個 session 統一成：
```
Session {
  session_id: str
  split: train/val/test
  label: 0/1   (只用於分析，不給 LLM)
  lines: List[str]   (原始 log 行)
}
```

### 1.3 MVP 完成判準
- 能印出：train/val/test session 數量、平均行數、anomaly ratio
- 隨機抽 3 個 session 看內容正常

---

## 2. 正規化層：Log Normalizer（RAG 成敗關鍵）

### 2.1 要做什麼
把 IP/UUID/HEX/path/數字等動態參數統一化，讓檢索「抓行為模式」而不是「抓參數」。

### 2.2 輸入/輸出
- Input：`lines: List[str]`
- Output：
  - `normalized_text: str`（把 lines join 後再 normalize）
  - （可選）`param_stats: dict`（例如 `<IPV4>` 出現次數，後續可做 complexity gating）

### 2.3 MVP 完成判準
- 抽樣前後對照：確保 `<IPV4>`, `<HEX>` 等替換成功
- normalize 後字串長度下降、重複模式更明顯

---

## 3. Screener 介面層：AllLinLog Inference Wrapper

### 3.1 要做什麼
把 AllLinLog 推論包成一個函式，輸出 logits 與 prediction。

### 3.2 輸入/輸出
- Input：`Session`
- Output：
```
ScreenerOutput {
  pred: 0/1
  logits: [l_norm, l_anom]
  prob: [p_norm, p_anom]
  margin: abs(p_anom - p_norm)
}
```

### 3.3 MVP 完成判準
- 在 test split 重算一次 F1，與你現有結果一致（BGL≈0.999、HDFS≈0.997）
- 能輸出每個 session 的 margin 分佈（為進階 gating 鋪路）

---

## 4. Evidence Store 建立：RAG Corpus Builder

### 4.1 要做什麼
把 **train split 的所有 sessions（normal + anomaly）** 轉成 evidence documents，建立可檢索索引。

### 4.2 Document 定義（建議）
每個 evidence doc：
```
EvidenceDoc {
  evidence_id: "E_train_<session_id>"
  session_id: <session_id>
  text: <normalized_text>
  meta: {label, length, dataset}  (label 不餵給 LLM)
}
```

### 4.3 關鍵規則（避免 reviewer 打槍）
- 只用 train 當 corpus
- retrieval 時排除同 session_id（保險起見）
- label 只做分析，不放進 prompt

### 4.4 MVP 完成判準
- evidence doc 數量 = train session 數量
- 隨機抽 evidence_doc 看 text 乾淨且可讀

---

## 5. Retriever：Top-k Evidence Retrieval

### 5.1 要做什麼
對每個要解釋的 anomaly session，用 normalized query 去 evidence corpus 找 top-k 相似證據。

### 5.2 最穩次序
- MVP：BM25（快、可解釋、可重現）
- 進階：dense/hybrid（當 ablation）

### 5.3 輸入/輸出
- Input：`query_normalized_text`, `k`
- Output：`List[EvidenceHit]`，包含 `evidence_id, score, text`

### 5.4 MVP 完成判準
- 抽樣 20 個 anomaly session，人工看 top-5 evidence 確實「像」（至少關鍵 error pattern 類似）
- 產出一個 retrieval 指標（例如：top-k 裡同 label 比例，只作分析）

---

## 6. Prompt 組裝：LLM Input Assembler（固定模板）

### 6.1 要做什麼
把 query session + top-k evidence 組成固定 prompt，要求 LLM 輸出 JSON trace schema。

### 6.2 Prompt 必含元素
- Task：為何這個 session 異常？（只談異常）
- Constraint：**每個 claim 必須引用 evidence_ids**
- Evidence block：E1..Ek（含 evidence_id 與文本）
- Output：JSON schema（固定欄位）

### 6.3 MVP 完成判準
- 用 5–10 筆樣本跑 LLM，JSON 可 parse、evidence_ids 存在且引用 E1..Ek

---

## 7. Trace Schema：Explanation JSON 定義（你的核心產物）

### 7.1 要做什麼
定義你論文中要展示的 schema，並在程式中嚴格驗證。

### 7.2 建議 schema（最小版）
```json
{
  "prediction": "anomaly",
  "claims": [
    {"claim": "...", "evidence_ids": ["E1","E3"]}
  ],
  "insufficient_evidence": false
}
```

### 7.3 MVP 完成判準
- JSON parse 成功率 > 95%（越高越好）
- evidence coverage：所有 claim 都有 evidence_ids（目標 100%）

---

## 8. Verifier：輕量可重現的 Faithfulness 檢查

### 8.1 要做什麼
不用第二個 LLM，先用 rule-based 做「最小可信度保護」。

### 8.2 MVP verifier 規則（先求穩）
- 檢查 evidence_ids 都存在於提供的 evidence 清單
- （可選）claim 中含有的關鍵詞/錯誤碼是否能在對應 evidence text 找到（regex/string match）

### 8.3 MVP 完成判準
- 產出 verifier pass rate
- 能列出 fail 的例子（供 Error Analysis）

---

## 9. Pipeline A：Baseline Explain-All（第一個可交付系統）

### 9.1 要做什麼
只對 `pred==anomaly` 的 sessions：
- retrieve top-k evidence
- 呼叫 LLM 產生 trace JSON
- verifier 檢查、存檔

### 9.2 完成判準（最重要）
- 能跑完整個 test split（或至少 1k sessions）
- 產出結果檔（jsonl/csv）
- 報告：
  - 平均 tokens/解釋
  - P50/P95 latency
  - evidence coverage、verifier pass rate
  - 抽樣 20 筆人工讀起來合理

---

## 10. Pipeline B：進階 Budgeted Explain（margin gating）

### 10.1 要做什麼
在 predicted anomalies 中，只解釋「低 margin」那一批（或固定 K%）。

### 10.2 設定方式（最簡單）
- 先收集所有 anomalies 的 margin
- 選 margin 最小的前 K%（例如 20%）進 Reasoner

### 10.3 完成判準
- 跟 Explain-All 比較：
  - tokens/latency 明顯下降
  - faithfulness 指標下降不大（或維持）
- 畫出 trade-off：K% vs faithfulness、K% vs cost

---

## 11. 評估與輸出（你之後寫論文最需要的資料）

程式跑完需自動產出：
- Detection：F1（Screener）
- Explanation：
  - JSON parse rate
  - Evidence coverage（目標 100%）
  - Verifier pass rate
- Cost：
  - trigger rate（Explain-All=anomaly ratio；Budgeted=K%×anomaly ratio）
  - avg tokens、P95 latency
- Ablation（可選）：top-k=1/3/5/10

---

## 12. Error Analysis（寫論文加分、也幫你 debug）

至少收集四類失敗案例：
- evidence 找不到（insufficient_evidence）
- evidence 找錯（retrieval mismatch）
- claim 很空洞（generic claim）
- verifier fail（引用不存在/不支援）

---

## 建議實作順序（最省力）
**Step 1：Normalizer → Evidence corpus（train）→ BM25 retrieval（top-5）**  
跑通後再接 LLM。
