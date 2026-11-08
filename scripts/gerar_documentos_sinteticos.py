r"""
Gera documentos societários sintéticos (PDF nativo) com dados fictícios (Faker pt_BR),
para complementar o dataset real de treino do NER (ver PRD.md secao 10).

Diferença fundamental em relação ao pipeline dos documentos reais (notebooks 01/02/03):
aqui o gabarito NÃO precisa de anotação via Gemini nem de alinhamento fuzzy — como o
texto é gerado por nós, sabemos exatamente que trecho é CPF/RG/NOME/ENDERECO desde a
inserção. O bbox de cada entidade é obtido com precisão exata via PyMuPDF
`page.search_for(...)` sobre o PDF já renderizado (mesma biblioteca usada no notebook 02
para PDF nativo), garantindo o mesmo formato de saída que os notebooks 03/06 esperam.

Cada documento sintético gera dois JSONs, no mesmo schema dos notebooks reais:
  - <saida>/03.extracao_ocr_local/<doc_id>.json   (tokens + bbox, como o notebook 02)
  - <saida>/04.gabarito_alinhado/sintetico/<doc_id>.json  (entidades + bbox, como o notebook 03)
Mais o PDF renderizado em <saida>/pdfs/<doc_id>.pdf, para auditoria visual.

Ruído deliberado (não rotulado): razão social, CNPJ, NIRE, cargo, valor de capital social,
datas — para o modelo aprender a NÃO marcar esses campos como PII de pessoa física
(risco de confusão PERSON/ORG documentado no PRD.md secao 8.1).

Uso:
    python scripts/gerar_documentos_sinteticos.py --n 500
    python scripts/gerar_documentos_sinteticos.py --n 500 --simular-scan
    python scripts/gerar_documentos_sinteticos.py --n 50 --seed 123 --saida data/documento_social_sintetico
"""

import argparse
import json
import math
import random
from pathlib import Path

import fitz  # PyMuPDF
from faker import Faker

# --------------------------------------------------------------------------------------
# Layout / render
# --------------------------------------------------------------------------------------

LARGURA_PAGINA, ALTURA_PAGINA = fitz.paper_size("a4")
MARGEM = 50
LARGURA_UTIL = LARGURA_PAGINA - 2 * MARGEM
FONTE_CORPO = "helv"
FONTE_TITULO = "hebo"
TAM_FONTE_CORPO = 11
TAM_FONTE_TITULO = 14
CHARS_POR_LINHA = 92  # estimativa grosseira para Helvetica 11pt em ~495pt de largura útil
ALTURA_LINHA = 14.5


def _estimar_altura(texto: str, chars_por_linha: int = CHARS_POR_LINHA) -> float:
    n_linhas = max(1, math.ceil(len(texto) / chars_por_linha))
    return n_linhas * ALTURA_LINHA + 8


class ConstrutorPDF:
    """Gerencia paginação manual de um documento PyMuPDF, com cursor vertical.

    Cada método de inserção de texto retorna o número da página (1-based) onde o
    texto acabou de ser colocado — quem chama usa isso pra saber em que `page` marcar
    as entidades daquele trecho.
    """

    def __init__(self):
        self.doc = fitz.open()
        self.pagina = None
        self.y = MARGEM
        self._nova_pagina()

    def _nova_pagina(self):
        self.pagina = self.doc.new_page(width=LARGURA_PAGINA, height=ALTURA_PAGINA)
        self.y = MARGEM

    @property
    def numero_pagina(self) -> int:
        return self.doc.page_count  # new_page já foi chamado, então é a página atual

    def _garantir_espaco(self, altura_necessaria: float):
        if self.y + altura_necessaria > ALTURA_PAGINA - MARGEM:
            self._nova_pagina()

    def titulo(self, texto: str) -> int:
        altura = TAM_FONTE_TITULO * 1.6 + 10
        self._garantir_espaco(altura)
        rect = fitz.Rect(MARGEM, self.y, LARGURA_PAGINA - MARGEM, self.y + altura)
        self.pagina.insert_textbox(rect, texto, fontsize=TAM_FONTE_TITULO, fontname=FONTE_TITULO, align=1)
        self.y += altura + 12
        return self.numero_pagina

    def paragrafo(self, texto: str) -> int:
        # `insert_textbox` com align=3 (justificado) NÃO desenha nada — nem parcialmente —
        # se o texto não couber inteiro no rect: só devolve um "sobra" negativo dizendo
        # quanto faltou. Por isso, se a estimativa inicial for insuficiente, cresce o rect
        # e tenta de novo (nada foi desenhado na tentativa anterior, então é seguro repetir).
        altura = _estimar_altura(texto)
        for _ in range(6):
            self._garantir_espaco(altura)
            rect = fitz.Rect(MARGEM, self.y, LARGURA_PAGINA - MARGEM, self.y + altura)
            # align=0 (esquerda), não justificado: justificado (align=3) estica o espaço
            # entre palavras pra preencher a linha, e isso faz o search_for() (usado depois
            # pra achar o bbox exato de cada entidade) devolver múltiplos retângulos picotados
            # pra uma mesma ocorrência em vez de um retângulo único — quebra o casamento
            # ocorrência->retângulo quando o mesmo texto aparece mais de uma vez no documento.
            sobra = self.pagina.insert_textbox(rect, texto, fontsize=TAM_FONTE_CORPO, fontname=FONTE_CORPO, align=0)
            if sobra >= 0:
                break
            altura += abs(sobra) + 12
        self.y += (altura - max(sobra, 0)) + 10
        return self.numero_pagina

    def espaco(self, altura: float = 12.0):
        self._garantir_espaco(altura)
        self.y += altura


# --------------------------------------------------------------------------------------
# Geração de dados fictícios (Faker) — pessoas e ruído (empresa/identificadores)
# --------------------------------------------------------------------------------------

TITULOS_PESSOA = ["", "Sr. ", "Sra. ", "Dr. ", "Dra. ", "Bel. "]  # "" = sem título, é a maioria
CARGOS_SOCIO = ["Sócio Administrador", "Sócio-Gerente", "Diretor Presidente", "Sócio Cotista", "Procurador"]
SUFIXOS_EMPRESA = ["LTDA", "EIRELI", "ME", "S.A.", "LTDA - ME"]


def gerar_pessoa(fake: Faker, rng: random.Random) -> dict:
    """Gera uma pessoa fictícia com todos os campos PII que o modelo precisa aprender."""
    titulo = rng.choice(TITULOS_PESSOA) if rng.random() < 0.25 else ""
    nome_base = fake.name()
    nome_completo = f"{titulo}{nome_base}"

    cpf = fake.cpf()

    rg_digitos = fake.rg()
    if rng.random() < 0.5 and len(rg_digitos) >= 8:
        # formata como XX.XXX.XXX-X (padrão comum, embora RG não tenha formato único nacional)
        rg_fmt = f"{rg_digitos[:2]}.{rg_digitos[2:5]}.{rg_digitos[5:8]}-{rg_digitos[8:]}"
    else:
        rg_fmt = rg_digitos

    logradouro = fake.street_address()
    bairro = fake.bairro()
    cidade = fake.city()
    uf = fake.state_abbr()
    cep = fake.postcode()
    endereco = f"{logradouro}, {bairro} - {cidade} - {uf}, CEP: {cep}"

    return {
        "nome": nome_completo,
        "cpf": cpf,
        "rg": rg_fmt,
        "endereco": endereco,
        "cargo": rng.choice(CARGOS_SOCIO),
    }


def gerar_ruido_empresa(fake: Faker, rng: random.Random) -> dict:
    """Campos de PESSOA JURÍDICA — nunca devem ser rotulados como PII de pessoa física
    (risco de confusão PERSON/ORG, ver PRD.md secao 8.1)."""
    razao_social = f"{fake.company()} {rng.choice(SUFIXOS_EMPRESA)}"
    cnpj = fake.cnpj()
    nire = "".join(rng.choice("0123456789") for _ in range(11))
    nire_fmt = f"{nire[:2]}.{nire[2:5]}.{nire[5:8]}-{nire[8:]}"
    capital_social = f"R$ {rng.randint(10, 5000)}.{rng.randint(0,999):03d},00"
    return {
        "razao_social": razao_social,
        "cnpj": cnpj,
        "nire": nire_fmt,
        "capital_social": capital_social,
        "data": fake.date_between(start_date="-10y", end_date="today").strftime("%d/%m/%Y"),
    }


# --------------------------------------------------------------------------------------
# Templates de documento — cada um retorna a lista de entidades marcadas (com página)
# --------------------------------------------------------------------------------------

def _marcar(entidades: list, texto: str, tipo: str, pagina: int):
    entidades.append({"text": texto, "type": tipo, "page": pagina})


def template_contrato_social(fake: Faker, rng: random.Random, b: ConstrutorPDF) -> list:
    entidades = []
    empresa = gerar_ruido_empresa(fake, rng)
    n_socios = rng.randint(2, 4)
    socios = [gerar_pessoa(fake, rng) for _ in range(n_socios)]

    p = b.titulo("CONTRATO SOCIAL")

    intro = (
        f"Pelo presente instrumento particular de contrato social, os abaixo assinados, "
        f"na qualidade de sócios, resolvem constituir uma sociedade empresária limitada, "
        f"sob a denominação de {empresa['razao_social']}, inscrita no CNPJ sob o "
        f"nº {empresa['cnpj']}, mediante as cláusulas e condições seguintes."
    )
    p = b.paragrafo(intro)

    for i, socio in enumerate(socios, start=1):
        texto_socio = (
            f"{i}) {socio['nome']}, {socio['cargo']}, portador do CPF nº {socio['cpf']} "
            f"e do RG nº {socio['rg']}, residente e domiciliado(a) em {socio['endereco']}."
        )
        p = b.paragrafo(texto_socio)
        _marcar(entidades, socio["nome"], "NOME", p)
        _marcar(entidades, socio["cpf"], "CPF", p)
        _marcar(entidades, socio["rg"], "RG", p)
        _marcar(entidades, socio["endereco"], "ENDERECO", p)

    p = b.paragrafo(
        f"CLÁUSULA PRIMEIRA - DO CAPITAL SOCIAL: o capital social é de {empresa['capital_social']}, "
        f"totalmente integralizado em moeda corrente nacional pelos sócios acima qualificados, "
        f"na proporção de suas respectivas quotas."
    )
    p = b.paragrafo(
        f"CLÁUSULA SEGUNDA - DA ADMINISTRAÇÃO: a administração da sociedade caberá a "
        f"{socios[0]['nome']}, já qualificado(a), que poderá praticar todos os atos "
        f"necessários à consecução dos objetivos sociais."
    )
    _marcar(entidades, socios[0]["nome"], "NOME", p)

    p = b.paragrafo(
        f"E por estarem justos e contratados, firmam o presente instrumento em {empresa['data']}, "
        f"perante a Junta Comercial competente, registrado sob NIRE nº {empresa['nire']}."
    )

    return entidades


def template_alteracao_contratual(fake: Faker, rng: random.Random, b: ConstrutorPDF) -> list:
    entidades = []
    empresa = gerar_ruido_empresa(fake, rng)
    n_socios = rng.randint(1, 3)
    socios = [gerar_pessoa(fake, rng) for _ in range(n_socios)]

    p = b.titulo(f"{rng.randint(1, 20)}ª ALTERAÇÃO CONTRATUAL")

    p = b.paragrafo(
        f"Os sócios de {empresa['razao_social']}, inscrita no CNPJ sob o nº {empresa['cnpj']}, "
        f"registrada na Junta Comercial sob NIRE nº {empresa['nire']}, resolvem alterar o "
        f"contrato social vigente conforme as cláusulas abaixo."
    )

    for socio in socios:
        texto = (
            f"{socio['nome']}, {socio['cargo']}, portador(a) do CPF nº {socio['cpf']}, "
            f"RG nº {socio['rg']}, residente em {socio['endereco']}, "
            f"resolve ingressar/retirar-se da sociedade nos termos desta alteração."
        )
        p = b.paragrafo(texto)
        _marcar(entidades, socio["nome"], "NOME", p)
        _marcar(entidades, socio["cpf"], "CPF", p)
        _marcar(entidades, socio["rg"], "RG", p)
        _marcar(entidades, socio["endereco"], "ENDERECO", p)

    p = b.paragrafo(
        f"Permanecem inalteradas as demais cláusulas do contrato social não expressamente "
        f"alteradas por este instrumento, firmado em {empresa['data']}."
    )
    return entidades


def template_requerimento_empresario(fake: Faker, rng: random.Random, b: ConstrutorPDF) -> list:
    entidades = []
    pessoa = gerar_pessoa(fake, rng)
    empresa = gerar_ruido_empresa(fake, rng)

    p = b.titulo("REQUERIMENTO DE EMPRESÁRIO")

    p = b.paragrafo(
        f"{pessoa['nome']}, {pessoa['cargo'].lower()}, portador(a) do CPF nº {pessoa['cpf']} "
        f"e do RG nº {pessoa['rg']}, residente e domiciliado(a) em {pessoa['endereco']}, "
        f"requer à Junta Comercial o registro de seus atos constitutivos como empresário "
        f"individual, sob a denominação de {empresa['razao_social']}."
    )
    _marcar(entidades, pessoa["nome"], "NOME", p)
    _marcar(entidades, pessoa["cpf"], "CPF", p)
    _marcar(entidades, pessoa["rg"], "RG", p)
    _marcar(entidades, pessoa["endereco"], "ENDERECO", p)

    p = b.paragrafo(
        f"Capital social declarado: {empresa['capital_social']}. "
        f"CNPJ requerido junto à Receita Federal, com sede no endereço acima informado."
    )

    p = b.paragrafo(f"Nestes termos, pede deferimento. Data: {empresa['data']}.")
    p = b.paragrafo(f"{pessoa['nome']}")
    _marcar(entidades, pessoa["nome"], "NOME", p)

    return entidades


def template_ata_assembleia(fake: Faker, rng: random.Random, b: ConstrutorPDF) -> list:
    entidades = []
    empresa = gerar_ruido_empresa(fake, rng)
    n_presentes = rng.randint(2, 5)
    presentes = [gerar_pessoa(fake, rng) for _ in range(n_presentes)]

    p = b.titulo("ATA DE ASSEMBLEIA GERAL")

    p = b.paragrafo(
        f"Aos {empresa['data']}, reuniram-se os sócios de {empresa['razao_social']} "
        f"(CNPJ nº {empresa['cnpj']}), em assembleia geral, com a presença dos seguintes membros:"
    )

    for socio in presentes:
        texto = f"{socio['nome']}, {socio['cargo']}, CPF nº {socio['cpf']}, RG nº {socio['rg']}."
        p = b.paragrafo(texto)
        _marcar(entidades, socio["nome"], "NOME", p)
        _marcar(entidades, socio["cpf"], "CPF", p)
        _marcar(entidades, socio["rg"], "RG", p)

    p = b.paragrafo(
        f"A presidência dos trabalhos foi assumida por {presentes[0]['nome']}, que declarou "
        f"instalada a assembleia, tendo sido deliberado e aprovado por unanimidade o balanço "
        f"patrimonial do exercício encerrado."
    )
    _marcar(entidades, presentes[0]["nome"], "NOME", p)

    p = b.paragrafo(
        f"Nada mais havendo a tratar, foi lavrada a presente ata, que segue assinada pelos "
        f"presentes e será registrada perante a Junta Comercial."
    )
    return entidades


def template_procuracao(fake: Faker, rng: random.Random, b: ConstrutorPDF) -> list:
    entidades = []
    outorgante = gerar_pessoa(fake, rng)
    outorgado = gerar_pessoa(fake, rng)

    p = b.titulo("PROCURAÇÃO")

    p = b.paragrafo(
        f"OUTORGANTE: {outorgante['nome']}, portador(a) do CPF nº {outorgante['cpf']} e "
        f"do RG nº {outorgante['rg']}, residente e domiciliado(a) em {outorgante['endereco']}."
    )
    _marcar(entidades, outorgante["nome"], "NOME", p)
    _marcar(entidades, outorgante["cpf"], "CPF", p)
    _marcar(entidades, outorgante["rg"], "RG", p)
    _marcar(entidades, outorgante["endereco"], "ENDERECO", p)

    p = b.paragrafo(
        f"OUTORGADO: {outorgado['nome']}, portador(a) do CPF nº {outorgado['cpf']} e "
        f"do RG nº {outorgado['rg']}, residente e domiciliado(a) em {outorgado['endereco']}."
    )
    _marcar(entidades, outorgado["nome"], "NOME", p)
    _marcar(entidades, outorgado["cpf"], "CPF", p)
    _marcar(entidades, outorgado["rg"], "RG", p)
    _marcar(entidades, outorgado["endereco"], "ENDERECO", p)

    p = b.paragrafo(
        f"PODERES: o(a) outorgante nomeia e constitui seu bastante procurador o(a) outorgado(a) "
        f"acima qualificado(a), para representá-lo(a) perante a Junta Comercial e demais órgãos "
        f"públicos, podendo praticar todos os atos necessários ao fiel cumprimento deste mandato."
    )
    return entidades


TEMPLATES = [
    template_contrato_social,
    template_alteracao_contratual,
    template_requerimento_empresario,
    template_ata_assembleia,
    template_procuracao,
]


# --------------------------------------------------------------------------------------
# Bbox exato via PyMuPDF search_for + extração de tokens no mesmo schema do notebook 02
# --------------------------------------------------------------------------------------

def extrair_tokens_por_pagina(doc: fitz.Document) -> list:
    paginas = []
    for i, pagina in enumerate(doc, start=1):
        palavras = pagina.get_text("words")  # (x0, y0, x1, y1, texto, block_no, line_no, word_no)
        tokens = [{"text": w[4], "bbox": [w[0], w[1], w[2], w[3]]} for w in palavras]
        paginas.append({"page": i, "tokens": tokens})
    return paginas


def resolver_bbox_entidades(doc: fitz.Document, entidades_planejadas: list) -> tuple:
    """Para cada entidade planejada (text/type/page), busca o(s) retângulo(s) exatos via
    `page.search_for()`. Como o texto é 100% conhecido (não é OCR real), o único motivo de
    uma entidade não bater é o texto ter sido cortado por quebra de página/overflow do
    textbox — casos assim ficam marcados como "unmatched", no mesmo espírito do notebook 03.
    """
    entidades_resolvidas = []
    nao_resolvidas = []

    # agrupa por (page, text) para não repetir a busca a cada ocorrência do mesmo texto
    por_pagina_texto = {}
    for e in entidades_planejadas:
        chave = (e["page"], e["text"])
        por_pagina_texto.setdefault(chave, []).append(e)

    cache_busca = {}
    for (pagina_num, texto), ocorrencias in por_pagina_texto.items():
        chave_cache = (pagina_num, texto)
        if chave_cache not in cache_busca:
            pagina = doc[pagina_num - 1]
            retangulos = pagina.search_for(texto)
            retangulos.sort(key=lambda r: (round(r.y0, 1), r.x0))
            cache_busca[chave_cache] = retangulos
        retangulos = cache_busca[chave_cache]

        # PyMuPDF devolve um retângulo por linha (texto que quebra linha gera >1 retângulo
        # para a MESMA ocorrência). Cada ocorrência planejada consome os retângulos na ordem
        # em que aparecem no documento.
        n_ocorrencias = len(ocorrencias)
        if n_ocorrencias == 0:
            continue
        if not retangulos:
            nao_resolvidas.extend(ocorrencias)
            continue

        if len(retangulos) == n_ocorrencias:
            # caso comum e não ambíguo: 1 retângulo por ocorrência (texto de uma linha só).
            grupos = [[r] for r in retangulos]
        else:
            # fallback pra texto que quebra em >1 linha (ex.: ENDERECO longo): distribui
            # os retângulos disponíveis em partes iguais entre as ocorrências. Só é exato
            # quando todas as ocorrências quebram em N linhas iguais; em caso de dúvida,
            # prefira ver isso como sinal para revisar o texto/layout do template.
            tamanho_grupo = max(1, len(retangulos) // n_ocorrencias)
            grupos = []
            for idx in range(n_ocorrencias):
                inicio = idx * tamanho_grupo
                fim = inicio + tamanho_grupo if idx < n_ocorrencias - 1 else len(retangulos)
                grupos.append(retangulos[inicio:fim])

        for e, grupo in zip(ocorrencias, grupos):
            if not grupo:
                nao_resolvidas.append(e)
                continue
            bbox = [[r.x0, r.y0, r.x1, r.y1] for r in grupo]
            entidades_resolvidas.append({**e, "bbox": bbox, "match_type": "exact"})

    return entidades_resolvidas, nao_resolvidas


# --------------------------------------------------------------------------------------
# Simulação opcional de "documento escaneado" (ruído + rotação leve + blur)
# --------------------------------------------------------------------------------------

def simular_scan(doc: fitz.Document, pasta_saida: Path, doc_id: str):
    import numpy as np
    from PIL import Image, ImageFilter

    rng_img = random.Random(hash(doc_id) & 0xFFFFFFFF)
    for i, pagina in enumerate(doc, start=1):
        pix = pagina.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        angulo = rng_img.uniform(-1.5, 1.5)
        img = img.rotate(angulo, expand=True, fillcolor=(255, 255, 255))

        if rng_img.random() < 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng_img.uniform(0.3, 0.9)))

        arr = np.asarray(img).astype(np.int16)
        ruido = np.random.default_rng(rng_img.randint(0, 2**31)).normal(0, 6, arr.shape).astype(np.int16)
        arr = np.clip(arr + ruido, 0, 255).astype("uint8")
        img_final = Image.fromarray(arr)

        destino = pasta_saida / f"{doc_id}_p{i}.jpg"
        img_final.save(destino, quality=82)


# --------------------------------------------------------------------------------------
# Loop principal
# --------------------------------------------------------------------------------------

def gerar_documento(fake: Faker, rng: random.Random, doc_id: str) -> tuple:
    template = rng.choice(TEMPLATES)
    builder = ConstrutorPDF()
    entidades_planejadas = template(fake, rng, builder)

    tokens_por_pagina = extrair_tokens_por_pagina(builder.doc)
    entidades_resolvidas, nao_resolvidas = resolver_bbox_entidades(builder.doc, entidades_planejadas)

    for e in nao_resolvidas:
        e_marcado = dict(e)
        e_marcado.update({"match_type": "unmatched", "reason": "texto não encontrado no PDF renderizado (overflow/paginação)"})
        entidades_resolvidas.append(e_marcado)

    for idx, e in enumerate(entidades_resolvidas):
        e.setdefault("occurrence_index", 1)
        e.setdefault("confidence", "synthetic")
        e.setdefault("notes", "gerado sinteticamente (Faker pt_BR)")

    extracao = {"document_id": doc_id, "pages": tokens_por_pagina}
    gabarito = {"document_id": doc_id, "annotated_by": "sintetico", "entities": entidades_resolvidas}

    return builder.doc, extracao, gabarito, len(nao_resolvidas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=500, help="Quantos documentos sintéticos gerar (padrão: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade (padrão: 42)")
    parser.add_argument(
        "--saida",
        default=None,
        help="Pasta de saída (padrão: data/documento_social_sintetico/ na raiz do projeto)",
    )
    parser.add_argument(
        "--simular-scan",
        action="store_true",
        help="Além do PDF nativo, salva uma versão rasterizada com ruído/rotação/blur simulando scan "
             "(para depois rodar pelo pipeline de OCR real, notebook 02, e testar robustez).",
    )
    parser.add_argument(
        "--prefixo-id",
        default="sintetico",
        help="Prefixo do document_id gerado, para nunca colidir com IDs de documentos reais (padrão: 'sintetico')",
    )
    args = parser.parse_args()

    if args.saida:
        pasta_saida = Path(args.saida)
    else:
        raiz = Path(__file__).resolve().parent.parent
        pasta_saida = raiz / "data" / "documento_social_sintetico"

    pasta_pdfs = pasta_saida / "pdfs"
    pasta_extracao = pasta_saida / "03.extracao_ocr_local"
    pasta_gabarito = pasta_saida / "04.gabarito_alinhado" / "sintetico"
    pasta_scans = pasta_saida / "scans_simulados"
    for p in (pasta_pdfs, pasta_extracao, pasta_gabarito):
        p.mkdir(parents=True, exist_ok=True)
    if args.simular_scan:
        pasta_scans.mkdir(parents=True, exist_ok=True)

    fake = Faker("pt_BR")
    Faker.seed(args.seed)
    rng = random.Random(args.seed)

    print(f"Gerando {args.n} documento(s) sintético(s) (seed={args.seed}) em {pasta_saida}")
    if args.simular_scan:
        print("Simulação de scan ATIVADA — vai gerar também imagens JPEG com ruído/rotação/blur.")

    contagem_tipos = {}
    total_nao_resolvidas = 0
    docs_com_problema = []

    for i in range(1, args.n + 1):
        doc_id = f"{args.prefixo_id}_{i:05d}"
        doc, extracao, gabarito, n_problemas = gerar_documento(fake, rng, doc_id)

        doc.save(str(pasta_pdfs / f"{doc_id}.pdf"))

        (pasta_extracao / f"{doc_id}.json").write_text(
            json.dumps(extracao, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pasta_gabarito / f"{doc_id}.json").write_text(
            json.dumps(gabarito, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if args.simular_scan:
            simular_scan(doc, pasta_scans, doc_id)

        for e in gabarito["entities"]:
            if e.get("match_type") == "unmatched":
                continue
            contagem_tipos[e["type"]] = contagem_tipos.get(e["type"], 0) + 1

        if n_problemas:
            total_nao_resolvidas += n_problemas
            docs_com_problema.append((doc_id, n_problemas))

        doc.close()

        if i % 50 == 0 or i == args.n:
            print(f"  {i}/{args.n} documentos gerados...")

    print(f"\nConcluído: {args.n} documento(s) sintético(s) gerado(s) em {pasta_saida}")
    print(f"Entidades por tipo: {contagem_tipos}")
    if total_nao_resolvidas:
        print(
            f"\n[AVISO] {total_nao_resolvidas} entidade(s) não resolvida(s) (overflow/paginação) "
            f"em {len(docs_com_problema)} documento(s) — ver match_type='unmatched' nos gabaritos."
        )
    else:
        print("\nTodas as entidades foram resolvidas com bbox exato (match_type='exact').")

    print(
        "\nPara treinar com esses dados, aponte notebooks/06.treinar_bertimbau.ipynb pra incluir "
        f"'{pasta_extracao}' e '{pasta_gabarito}' junto com os documentos reais."
    )


if __name__ == "__main__":
    main()
