# 🧬 HER-2/neu Q&A Chatbot

A document-grounded chatbot that answers questions about HER-2/neu oncogene amplification and prognosis in breast cancer, using a [research paper](https://www.researchgate.net/profile/Gary-Clark/publication/19364043_Slamon_DJ_Clark_GM_Wong_SG_Levin_WJ_Ullrich_A_McGuire_WLHuman_breast_cancer_correlation_of_relapse_and_survival_with_amplification_of_the_HER-2neu_oncogene_Science_Wash_DC_235_177-182/links/0046352b85f241a532000000/Slamon-DJ-Clark-GM-Wong-SG-Levin-WJ-Ullrich-A-McGuire-WLHuman-breast-cancer-correlation-of-relapse-and-survival-with-amplification-of-the-HER-2-neu-oncogene-Science-Wash-DC-235-177-182.pdf) an open-source LLM (BioGPT). The chatbot supports continuous improvement via feedback logging, automated evaluation, and human-in-the-loop correction.

---

## 🚀 Features

- 🔍 Answers biomedical questions from a specific research paper (RAG + BioGPT)
- 📄 Built on a PDF publication from *Science, 1987* (Slamon et al.)
- 👍 Accepts thumbs-up/down feedback from users
- 🧪 Logs and evaluates semantic accuracy using BERTScore and cosine similarity
- 🔁 Supports continuous improvement via labeled corrections and versioning

---

## 🛠️ Technologies Used

- Python, Streamlit, LangChain
- `microsoft/BioGPT-Large` for biomedical Q&A
- FAISS for semantic retrieval
- `bert-score`, `sentence-transformers` for evaluation
- Remote logging via `.streamlit/secrets.toml`

---

## 💬 How to Run

### 📍 Local

```bash
git clone https://github.com/Arjithab/Humana-QA-chatbot.git
cd Humana-QA-chatbot

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

streamlit run run_chatbot.py

