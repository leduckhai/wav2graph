# Inductive Learning
## Node classification
python scripts/wav2graph/node_inductive_learning.py --config config/node_inductive_learning.yaml --embedding random
python scripts/wav2graph/node_inductive_learning.py --config config/node_inductive_learning.yaml --embedding vinai/phobert-base
python scripts/wav2graph/node_inductive_learning.py --config config/link_inductive_learning.yaml --embedding Alibaba-NLP/gte-large-en-v1.5
python scripts/wav2graph/node_inductive_learning.py --config config/link_inductive_learning.yaml --embedding Alibaba-NLP/gte-Qwen2-7B-instruct

## Link prediction
python scripts/wav2graph/link_inductive_learning.py --config config/link_inductive_learning.yaml --embedding random
python scripts/wav2graph/link_inductive_learning.py --config config/link_inductive_learning.yaml --embedding vinai/phobert-base
python scripts/wav2graph/link_inductive_learning.py --config config/link_inductive_learning.yaml --embedding Alibaba-NLP/gte-large-en-v1.5
python scripts/wav2graph/link_inductive_learning.py --config config/link_inductive_learning.yaml --embedding Alibaba-NLP/gte-Qwen2-7B-instruct

# Transductive Learning
## Node classification
python scripts/wav2graph/node_transductive_learning.py --config config/node_transductive_learning.yaml --embedding random
python scripts/wav2graph/node_transductive_learning.py --config config/node_transductive_learning.yaml --embedding vinai/phobert-base
python scripts/wav2graph/node_transductive_learning.py --config config/link_transductive_learning.yaml --embedding Alibaba-NLP/gte-large-en-v1.5
python scripts/wav2graph/node_transductive_learning.py --config config/link_transductive_learning.yaml --embedding Alibaba-NLP/gte-Qwen2-7B-instruct

## Link prediction
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning.yaml --embedding random
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning.yaml --embedding vinai/phobert-base
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning.yaml --embedding Alibaba-NLP/gte-large-en-v1.5
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning.yaml --embedding Alibaba-NLP/gte-Qwen2-7B-instruct

# ASR - Example: ASR 28.8 (Modify the config file accordingly)
## Node classification
python scripts/wav2graph/node_transductive_learning.py --config config/node_transductive_learning_asr.yaml --embedding random --type asr
python scripts/wav2graph/node_transductive_learning.py --config config/node_transductive_learning_asr.yaml --embedding vinai/phobert-base --type asr
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning_asr.yaml --embedding Alibaba-NLP/gte-large-en-v1.5 --type asr
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning_asr.yaml --embedding Alibaba-NLP/gte-Qwen2-7B-instruct --type asr

## Link prediction
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning_asr.yaml --embedding random --type asr
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning_asr.yaml --embedding vinai/phobert-base --type asr
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning_asr.yaml --embedding Alibaba-NLP/gte-large-en-v1.5 --type asr
python scripts/wav2graph/link_transductive_learning.py --config config/link_transductive_learning_asr.yaml --embedding Alibaba-NLP/gte-Qwen2-7B-instruct --type asr
