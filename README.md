# ML-Based Network Intrusion Detection System (NIDS)

Detecting malicious network activity (port scans, DDoS, brute-force, etc.) by applying
machine learning to network traffic data. Built to bridge a data science / analytics
background into network security & SOC-style detection work.

## Why this project
Security Operations Centers (SOCs) and detection engineering teams drown in network and
log data. This project applies ML/analytics to that data to automatically flag intrusions —
the intersection of data science and network defense.

## Roadmap
- [ ] **Phase 1 – Foundations + first model** (NSL-KDD): understand network flows & attack
      categories, build and evaluate a first classifier.
- [ ] **Phase 2 – Realistic data** (CICIDS2017): modern attacks, flow features, class imbalance.
- [ ] **Phase 3 – Detection engineering**: precision/recall tuning, confusion analysis,
      false-positive cost.
- [ ] **Phase 4 – Live capture**: sniff real traffic with scapy, extract features, score it live.
- [ ] **Phase 5 – Portfolio polish**: dashboard, writeup, GitHub README.
- [ ] **Phase 6 – Cloud integration ($0)**: deploy the detector as a service and ingest
      real cloud security logs (public AWS VPC Flow Logs / CloudTrail samples), using
      LocalStack + free tiers only. No paid cloud resources.
- [ ] **Phase 7 – SOC Copilot (RAG, $0)**: build a Retrieval-Augmented Generation
      assistant that explains each flagged alert in plain English using a security
      knowledge base (MITRE ATT&CK, attack descriptions, remediation). Fully local:
      Ollama (LLM) + sentence-transformers (embeddings) + Chroma/FAISS (vector store).
      Teaches embeddings, chunking, retrieval, prompt augmentation.

## Structure
```
data/raw/         # downloaded datasets (gitignored)
data/processed/   # cleaned/feature-engineered data
notebooks/        # exploratory analysis
src/              # reusable pipeline code
models/           # trained models
reports/          # figures, findings, writeups
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data
The NSL-KDD dataset is not committed (it's large). Download it into `data/raw/`:
```bash
mkdir -p data/raw && cd data/raw
BASE="https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master"
curl -sSL -o "KDDTrain+.txt" "$BASE/KDDTrain+.txt"
curl -sSL -o "KDDTest+.txt"  "$BASE/KDDTest+.txt"
```

## Concepts learned
Networking: TCP/IP, ports, protocols (TCP/UDP/ICMP), flows vs packets.
Security: intrusion types, detection evaluation, false-positive tradeoffs.
