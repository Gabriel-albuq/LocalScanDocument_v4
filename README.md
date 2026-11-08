# LocalScanDocument — Detecção e Mascaramento Local de PII em Documentos Societários

Projeto final do curso de especialização — **Centro de Informática (CIn) / UFPE**
Autor: Gabriel de Albuquerque Araújo

- 📄 **Relatório:** [`novo_artigo/artigo.pdf`](novo_artigo/artigo.pdf) (fonte LaTeX em [`novo_artigo/artigo.tex`](novo_artigo/artigo.tex))
- 🎥 **Vídeo (≤ 7 min):** _link a adicionar_

---

## O problema

Analisar contratos sociais e alterações contratuais (KYC/KYB) com um LLM em nuvem normalmente exige enviar o documento inteiro ao provedor, o que expõe **CPF, RG, nome e endereço** dos sócios a terceiros, com risco sob a LGPD.

## A solução: arquitetura híbrida *Edge-Cloud*

O documento original **nunca sai do ambiente local**:

```
PDF / imagem ──► Extração local ──► NER local ──► Mascaramento ──► LLM em nuvem ──► Desmascaramento ──► Resposta
                (PyMuPDF +          (BERTimbau,     (retângulo +     (Gemini, recebe     (substituição      ao usuário
                 PaddleOCR,          4 cabeças,      marcador         só a página        local pelo mapa
                 tokens + bbox)      janela          [NOME_A3K9F])    mascarada +        de marcadores)
                                     deslizante)                      a pergunta)
```

1. **Extração**: texto nativo via PyMuPDF; páginas escaneadas via PaddleOCR. A saída é uma lista de *tokens* com *bounding box*.
2. **Detecção**: o BERTimbau classifica cada *token* no esquema BIO para 4 entidades: `CPF`, `RG`, `NOME`, `ENDERECO`.
3. **Mascaramento**: cada entidade é coberta na imagem da página por um marcador aleatório (`[TIPO_código]`).
4. **Nuvem**: o Gemini responde à pergunta do usuário vendo apenas a página mascarada.
5. **Desmascaramento**: os marcadores da resposta são trocados localmente pelos valores reais.

## Principais resultados

### Detector local (NER) — evolução ao longo de 6 ciclos

| Modelo | O que mudou | F1 | Recall |
|---|---|---|---|
| 1 | 500 docs reais (473 usáveis) | 0,777 | 0,837 |
| 2 | + 500 sintéticos | 0,850 | 0,895 |
| 3 | 1000 reais + 500 sintéticos (**memorização do sintético**: F1 real 0,72 vs. sintético 1,00) | 0,850 | 0,895 |
| 4 | só 1000 reais (controle) | 0,816 | 0,875 |
| 5 | + regex, propagação por texto e deduplicação do gabarito (holdout) | 0,839 | 0,882 |
| **6** | **janela deslizante + 4 cabeças de classificação (holdout)** | **0,873** | **0,927** |

O recall foi a métrica principal, porque deixar um dado sensível sem máscara (falso negativo) custa mais caro do que mascarar algo à toa. Recall por entidade no Modelo 6: CPF 0,983 · RG 0,926 · NOME 0,919 · ENDERECO 0,846.

### Pipeline em nuvem — 28 documentos reais do holdout, 3 perguntas informativas (84 casos por modelo)

| Métrica | gemini-2.5-flash | gemini-2.5-pro |
|---|---|---|
| Erro sem máscara (*baseline*) | 22,6% | 19,0% |
| Erro com o pipeline (após desmascarar) | 32,1% | **16,7%** |
| — causado pelo processo de mascaramento | 20,2% | 8,3% |
| Pergunta de *guardrail* ("Gabriel é sócio?") | 0% de erro | 0% de erro |

Nenhum valor real de CPF, RG, nome ou endereço chega ao provedor de LLM.

## Estrutura do repositório

```
├── notebooks/
│   ├── bert/                       # pipeline principal, executar na ordem numérica
│   │   ├── 00  gerar documentos sintéticos
│   │   ├── 01  gerar ground truth com Gemini (lendo o PDF)
│   │   ├── 02  extrair texto local (PyMuPDF + PaddleOCR)   ← kernel .venv_ocr
│   │   ├── 02b extrair um arquivo avulso                     ← kernel .venv_ocr
│   │   ├── 03  alinhar ground truth ↔ tokens (fuzzy, rapidfuzz)
│   │   ├── 04/05 baselines spaCy / GLiNER (desativados, "- off")
│   │   ├── 06  treinar BERTimbau (10 cenários de hiperparâmetro)
│   │   ├── 07–09 ground truth, extração e alinhamento do holdout
│   │   ├── 10  métricas no holdout (IoU ≥ 0,30 por bounding box)
│   │   ├── 11  verificação manual dos resultados
│   │   └── 12  inferência em documentos sem gabarito
│   ├── bert_4_cabecas/             # Modelo 6: 4 cabeças + janela deslizante (06, 10, 11, 12)
│   ├── rerun_comparacao_holdout/   # reavaliação dos modelos 1–4 e validação do pipeline em nuvem
│   └── pipeline_completo.ipynb     # demonstração de ponta a ponta (extração → NER → máscara → Gemini → resposta)
├── scripts/
│   ├── extrair_ocr_worker.py              # worker de OCR rodado em subprocesso no .venv_ocr
│   ├── gerar_documentos_sinteticos.py     # documentos societários fictícios (Faker pt_BR)
│   ├── pos_processamento_ner.py           # regex de assinatura digital + propagação por texto
│   ├── selecionar_amostra_documentos.py   # amostragem aleatória do treino
│   └── selecionar_amostra_teste_holdout.py
├── prompts/                        # prompts e schema do ground truth; instrução de mascaramento enviada ao LLM
├── novo_artigo/                    # relatório final (LaTeX + PDF) e slides
├── PRD.md                          # requisitos e decisões técnicas do projeto
├── requirements.txt                # ambiente geral (torch, transformers, Gemini...)
└── requirements_ocr.txt            # ambiente isolado de OCR (PaddleOCR)
```

## Como reproduzir

O projeto usa **dois ambientes Python 3.10 separados**. O PaddlePaddle-GPU e o PyTorch trazem versões diferentes de cuDNN e não podem ser carregados no mesmo processo no Windows (`WinError 127`).

```bash
# 1) Ambiente geral (treino, inferência, Gemini)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2) Ambiente de OCR: instalar antes o wheel do paddlepaddle-gpu indicado no topo de requirements_ocr.txt
python -m venv .venv_ocr
.venv_ocr\Scripts\activate
pip install -r requirements_ocr.txt
```

Copie `.env.example` para `.env` e preencha `GEMINI_API_KEY`.

### Dados e modelos

- `data/` **não está no repositório**: são documentos societários reais com dados pessoais (LGPD). Os documentos sintéticos podem ser gerados com `scripts/gerar_documentos_sinteticos.py` (ou com o notebook `00`).
- `models/` **não está no repositório** por causa do tamanho (pesos dos 6 ciclos × 10 cenários).

## Referência

Inspirado na metodologia do *framework* **LegalGuardian** (arXiv:2501.10915), estendida para documentos reais (PDF nativo, escaneado e imagem) com mascaramento espacial por *bounding box*.
