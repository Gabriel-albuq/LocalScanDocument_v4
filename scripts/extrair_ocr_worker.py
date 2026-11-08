r"""Worker de extração (nativo PyMuPDF + PaddleOCR) para UM arquivo avulso.

Roda SÓ dentro do venv isolado de OCR (`.venv_ocr`, kernel `locscan-ocr`) -- paddleocr/paddlepaddle
não estão instalados no venv geral, e não podem ser importados no mesmo processo que
torch/transformers (conflito de cuDNN documentado em requirements_ocr.txt, WinError 127).

Por isso este script é feito para ser chamado como um PROCESSO SEPARADO (via subprocess,
usando o interpretador Python do `.venv_ocr` explicitamente) a partir de
`scripts/pipeline_completo.py`, que roda no venv geral -- nunca importe este módulo direto de
um processo que já carregou torch.

Mesma lógica de `notebooks/bert/02.extrair_texto_ocr_local.ipynb` e
`notebooks/bert/02b.extrair_um_arquivo_avulso.ipynb` (copiada, não importada -- mesma decisão
de autocontenção usada no resto do projeto para os notebooks; aqui o motivo de existir como
script à parte é justamente permitir a chamada via subprocess).

Uso:
    <venv_ocr>/Scripts/python.exe scripts/extrair_ocr_worker.py <arquivo_entrada> <json_saida>

Argumentos:
    arquivo_entrada  PDF ou imagem (.jpg/.jpeg/.png/.tif/.tiff/.bmp)
    json_saida       Caminho onde salvar o JSON de extração (mesmo formato usado pelos
                      notebooks 02/08/12: {document_id, source_file, pages: [{page, method,
                      tokens: [{text, bbox}]}]})
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

DPI_RENDER = 300
MIN_CHARS_TEXTO_NATIVO = 20
MIN_SPREAD_VERTICAL_TEXTO_NATIVO = 0.30
MIN_RATIO_LEGIVEL_TEXTO_NATIVO = 0.70

EXTENSOES_PDF = {".pdf"}
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def corrigir_inclinacao(img_cinza: np.ndarray) -> np.ndarray:
    """Detecta o ângulo dominante do texto via Hough e rotaciona para nivelar a página."""
    bordas = cv2.Canny(img_cinza, 50, 150, apertureSize=3)
    linhas = cv2.HoughLinesP(bordas, 1, np.pi / 180, threshold=200, minLineLength=img_cinza.shape[1] // 4, maxLineGap=20)

    if linhas is None or len(linhas) == 0:
        return img_cinza

    angulos = []
    for linha in linhas:
        x1, y1, x2, y2 = linha[0]
        angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angulo < 45:
            angulos.append(angulo)

    if not angulos:
        return img_cinza

    angulo_mediano = float(np.median(angulos))
    if abs(angulo_mediano) < 0.3:
        return img_cinza

    h, w = img_cinza.shape[:2]
    centro = (w // 2, h // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo_mediano, 1.0)
    return cv2.warpAffine(img_cinza, matriz, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocessar_imagem(img_bgr: np.ndarray) -> np.ndarray:
    """Pipeline determinístico: escala de cinza -> deskew -> denoise moderado -> threshold adaptativo."""
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cinza = corrigir_inclinacao(cinza)
    cinza = cv2.fastNlMeansDenoising(cinza, h=7, templateWindowSize=7, searchWindowSize=21)
    binarizada = cv2.adaptiveThreshold(
        cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=15
    )
    return binarizada


def _ratio_caracteres_legiveis(texto: str) -> float:
    if not texto:
        return 0.0
    validos = sum(1 for c in texto if c.isalpha() or c.isdigit() or c.isspace() or c in ".,;:-/()ºª%")
    return validos / len(texto)


def tem_texto_nativo(pagina) -> bool:
    texto = pagina.get_text().strip()
    if len(texto) < MIN_CHARS_TEXTO_NATIVO:
        return False
    if _ratio_caracteres_legiveis(texto) < MIN_RATIO_LEGIVEL_TEXTO_NATIVO:
        return False
    palavras = pagina.get_text("words")
    if not palavras:
        return False
    altura_pagina = pagina.rect.height
    if altura_pagina <= 0:
        return True
    ys = [w[1] for w in palavras]
    spread_relativo = (max(ys) - min(ys)) / altura_pagina
    return spread_relativo >= MIN_SPREAD_VERTICAL_TEXTO_NATIVO


def extrair_pagina_nativa(pagina) -> list:
    palavras = pagina.get_text("words")
    tokens = []
    for x0, y0, x1, y1, texto, *_ in palavras:
        texto = texto.strip()
        if texto:
            tokens.append({"text": texto, "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]})
    return tokens


def _box_para_bbox(box) -> list:
    box = np.asarray(box, dtype=float)
    if box.ndim == 1:
        x0, y0, x1, y1 = box[:4]
        return [float(x0), float(y0), float(x1), float(y1)]
    xs = box[:, 0]
    ys = box[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def mesclar_tokens_fragmentados(tokens: list) -> list:
    if not tokens:
        return tokens
    tokens_ordenados = sorted(tokens, key=lambda t: (round(t["bbox"][1], 0), t["bbox"][0]))
    mesclados = [dict(tokens_ordenados[0])]
    for tok in tokens_ordenados[1:]:
        anterior = mesclados[-1]
        altura_ref = max(anterior["bbox"][3] - anterior["bbox"][1], 1.0)
        mesma_linha = abs(tok["bbox"][1] - anterior["bbox"][1]) < altura_ref * 0.5
        gap = tok["bbox"][0] - anterior["bbox"][2]
        fragmento_isolado = len(tok["text"]) <= 2 and not tok["text"][:1].isupper()
        if mesma_linha and fragmento_isolado and 0 <= gap < altura_ref * 0.35:
            anterior["text"] = anterior["text"] + tok["text"]
            anterior["bbox"][2] = tok["bbox"][2]
            anterior["bbox"][1] = min(anterior["bbox"][1], tok["bbox"][1])
            anterior["bbox"][3] = max(anterior["bbox"][3], tok["bbox"][3])
        else:
            mesclados.append(dict(tok))
    return mesclados


def _carregar_ocr():
    """Import tardio -- só existe no venv isolado; falha alto e claro se rodado no venv errado."""
    import paddle
    from paddleocr import PaddleOCR

    cuda_disponivel = paddle.device.is_compiled_with_cuda()
    n_gpus = paddle.device.cuda.device_count() if cuda_disponivel else 0
    print(f"Paddle compilado com CUDA: {cuda_disponivel} | GPUs detectadas: {n_gpus}", file=sys.stderr)

    return PaddleOCR(
        device="gpu:0",
        lang="pt",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
        enable_mkldnn=False,
    )


def _dpi_efetivo_pagina(pagina, dpi_padrao: int = DPI_RENDER, cobertura_minima: float = 0.75) -> int:
    """Calcula o DPI que reproduz a resolução NATIVA da imagem escaneada dominante da página, em
    vez de sempre usar `dpi_padrao` fixo. Achado 2026-09-21: alguns PDFs escaneados declaram um
    MediaBox bem maior que o conteúdo real (ex.: scan A4 dentro de um MediaBox de ~21x30
    polegadas -- "Constituição 12-09-1996.pdf" do holdout). Nesses casos, `get_pixmap(dpi=300)`
    faz o PyMuPDF esticar/interpolar a imagem original (resolução fixa) sobre uma área bem maior
    que a real, produzindo uma imagem "borrada" com MENOS densidade de informação por área do
    que o DPI nominal sugere -- e o OCR sai pior nessas páginas. Usar o DPI calculado a partir da
    proporção real (pixels nativos da imagem / tamanho do bbox onde ela está desenhada) evita
    essa interpolação: quando o resultado é MENOR que `dpi_padrao`, ainda assim reproduz a
    imagem de forma mais fiel (praticamente 1:1, sem esticar) do que insistir em 300 DPI sobre
    uma página maior que o conteúdo real."""
    try:
        # Regressão real encontrada 2026-09-21 (documento "Alteração Consolidada 102019.pdf",
        # páginas 2/5/6/7): páginas com uma imagem de fundo quase em branco (papel de segurança/
        # marca d'água) cobrindo quase toda a área, mas com o conteúdo de verdade em TEXTO NATIVO
        # por cima -- calcular um DPI baseado só na imagem, nesse caso, é inofensivo aqui (esta
        # função só decide o DPI do get_pixmap, que sempre compõe texto+imagem corretamente), mas
        # ainda assim não faz sentido recalcular DPI a partir de uma imagem de fundo decorativa
        # quando a página tem texto nativo de verdade -- usa o DPI padrão nesse caso.
        if len(pagina.get_text().strip()) >= 20:
            return dpi_padrao
        area_pagina = pagina.rect.width * pagina.rect.height
        if area_pagina <= 0:
            return dpi_padrao
        melhor_bbox, melhor_area, melhor_largura_px = None, 0, None
        for info in pagina.get_image_info(xrefs=True):
            bbox = info.get("bbox")
            if not bbox:
                continue
            area_imagem = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area_imagem > melhor_area:
                melhor_area, melhor_bbox, melhor_largura_px = area_imagem, bbox, info.get("width")
        if melhor_bbox is None or not melhor_largura_px or (melhor_area / area_pagina) < cobertura_minima:
            return dpi_padrao
        largura_bbox_pt = melhor_bbox[2] - melhor_bbox[0]
        if largura_bbox_pt <= 0:
            return dpi_padrao
        return max(72, round((melhor_largura_px / largura_bbox_pt) * 72))
    except Exception:
        return dpi_padrao


def extrair_pagina_ocr(ocr, pagina, dpi: int = DPI_RENDER) -> list:
    dpi = _dpi_efetivo_pagina(pagina, dpi)
    pix = pagina.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if pix.n == 4 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_processada = preprocessar_imagem(img_bgr)
    img_ocr = cv2.cvtColor(img_processada, cv2.COLOR_GRAY2BGR)

    resultado = ocr.predict(img_ocr)
    if not resultado:
        return []
    r = resultado[0]

    escala = 72.0 / dpi
    tokens = []
    for palavras_linha, boxes_linha in zip(r.get("text_word", []), r.get("text_word_boxes", [])):
        tokens_linha = []
        for palavra, box in zip(palavras_linha, boxes_linha):
            palavra = palavra.strip()
            if not palavra:
                continue
            x0, y0, x1, y1 = _box_para_bbox(box)
            tokens_linha.append({
                "text": palavra,
                "bbox": [round(x0 * escala, 2), round(y0 * escala, 2), round(x1 * escala, 2), round(y1 * escala, 2)],
            })
        tokens.extend(mesclar_tokens_fragmentados(tokens_linha))
    return tokens


def processar_pdf(ocr, caminho: Path) -> dict:
    doc = pymupdf.open(str(caminho))
    paginas_saida = []
    for i, pagina in enumerate(doc, start=1):
        if tem_texto_nativo(pagina):
            tokens = extrair_pagina_nativa(pagina)
            metodo = "nativo"
        else:
            tokens = extrair_pagina_ocr(ocr, pagina)
            metodo = "ocr"
        paginas_saida.append({"page": i, "method": metodo, "tokens": tokens})
    doc.close()
    return {"document_id": caminho.stem, "source_file": caminho.name, "pages": paginas_saida}


def processar_imagem(ocr, caminho: Path) -> dict:
    img_bgr = cv2.imread(str(caminho))
    if img_bgr is None:
        raise ValueError(f"Não consegui abrir a imagem: {caminho}")
    img_processada = preprocessar_imagem(img_bgr)
    img_ocr = cv2.cvtColor(img_processada, cv2.COLOR_GRAY2BGR)

    resultado = ocr.predict(img_ocr)
    tokens = []
    if resultado:
        r = resultado[0]
        for palavras_linha, boxes_linha in zip(r.get("text_word", []), r.get("text_word_boxes", [])):
            tokens_linha = []
            for palavra, box in zip(palavras_linha, boxes_linha):
                palavra = palavra.strip()
                if not palavra:
                    continue
                x0, y0, x1, y1 = _box_para_bbox(box)
                tokens_linha.append({"text": palavra, "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]})
            tokens.extend(mesclar_tokens_fragmentados(tokens_linha))
    return {"document_id": caminho.stem, "source_file": caminho.name, "pages": [{"page": 1, "method": "ocr", "tokens": tokens}]}


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)

    caminho = Path(sys.argv[1])
    caminho_saida = Path(sys.argv[2])
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    extensao = caminho.suffix.lower()
    if extensao not in EXTENSOES_PDF | EXTENSOES_IMAGEM:
        raise ValueError(f"Extensão não suportada: {extensao} (esperado PDF ou {EXTENSOES_IMAGEM})")

    print(f"Carregando PaddleOCR...", file=sys.stderr)
    ocr = _carregar_ocr()

    if extensao in EXTENSOES_PDF:
        resultado = processar_pdf(ocr, caminho)
    else:
        resultado = processar_imagem(ocr, caminho)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    caminho_saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    n_tokens = sum(len(p["tokens"]) for p in resultado["pages"])
    metodos = {p["page"]: p["method"] for p in resultado["pages"]}
    print(f"{caminho.name}: {len(resultado['pages'])} pagina(s), {n_tokens} token(s) | metodos: {metodos}", file=sys.stderr)
    print(f"Extracao salva em: {caminho_saida}", file=sys.stderr)


if __name__ == "__main__":
    main()
