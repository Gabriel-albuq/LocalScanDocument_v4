r"""Pós-processamento heurístico das entidades previstas pelo BERTimbau (notebook 06/07/09).

Correções específicas para os padrões de erro encontrados na investigação de
2026-09-11 (ver anotacoes/anotacoes.txt) sobre o modelo `bertimbau-pii-v1-cenarios-sem-sinteticos`:

1. `extrair_nome_assinatura_digital`: o padrão "ASSINADO DIGITALMENTE POR: <CPF>-<NOME>"
   se repete em toda página de vários contratos (assinatura eletrônica da plataforma) e o
   modelo NUNCA captura esse NOME — porque o CPF e o nome saem fundidos em um único token
   pela extração OCR/nativa (ex.: "11638817758-DAVID"), quebrando a tokenização de subpalavras
   do BERT. Como o padrão é sempre no mesmo formato fixo, um regex determinístico recupera
   esses casos sem precisar retreinar o modelo.

2. `estender_boundary_nome`: quando o modelo prevê só um pedaço de um nome longo (4-5
   palavras) — ex. previu "ARGEMIRO" onde o gold é "ARGEMIRO GILBERTO SILVEIRA DOS SANTOS" —
   estende a entidade absorvendo tokens vizinhos em maiúsculas/Title Case, até um limite
   de palavras, para reduzir o erro de fronteira (boundary) sem alterar o tipo previsto.

3. `propagar_entidades_por_texto_identico`: quando o mesmo texto exato de uma entidade já
   detectada em algum lugar do documento aparece de novo em outra página/posição sem ter
   sido marcado (ex.: "Marinely de Paula Bomfim" reconhecido como NOME nas páginas 2/8/9/10
   de um documento mas não na página 1, mesmo nome de autoridade repetido em rodapé de
   certificação), propaga a marcação para as demais ocorrências dentro do mesmo documento.
   Ver notebook 07: o recall baixo de NOME é majoritariamente "FN_perdida" (o modelo não viu
   nada) concentrado nesse tipo de repetição — repropagar o que já foi visto em outro lugar
   do mesmo documento recupera esses casos sem precisar retreinar.

   Decisão de produto (2026-09-15): a prioridade é CONSISTÊNCIA de marcação entre páginas
   idênticas, não só recall — se um texto é marcado em uma página, ele deve ficar marcado
   (com a fronteira completa) em toda página onde o mesmo texto se repetir identicamente,
   mesmo quando a marcação-fonte for, ela mesma, um falso positivo (ex.: endereço de
   escritório de contabilidade repetido em rodapé de várias páginas — mascarar a mais é
   aceitável; uma marcação inconsistente entre páginas idênticas do MESMO documento é, na
   prática, um vazamento de mascaramento). Por isso a função agora também SUBSTITUI uma
   entidade existente que cobre só uma PARTE do span canônico (ex.: o modelo previu só "302"
   numa página onde o texto completo do carimbo é "AV. BIAS FORTES, 932 CONJ. 302 [...]")
   pelo span completo, em vez de desistir daquela página só porque já havia algo ali.

4. `mesclar_previsoes_por_janela`: BERTimbau tem limite físico de 512 posições. Uma página
   de qualificação societária com dois sócios facilmente passa de 400 palavras, que viram
   ~1000-1200 subtokens depois da tokenização em wordpieces — o texto que sobra depois do
   corte de 512 é truncado SILENCIOSAMENTE (vira token especial de padding/corte, nunca chega
   a ser classificado). É a causa raiz encontrada em 2026-09-15 para o caso real do documento
   MINAS CAL: o endereço do primeiro sócio (Elizabeth) cai antes do corte e é reconhecido; o
   endereço do segundo sócio (Sérgio), que vem depois no texto, cai além da posição 512 e
   nunca é visto pelo modelo — não é erro de classificação, o texto simplesmente não existe
   mais na entrada. Divide a página em janelas deslizantes com sobreposição quando ultrapassa
   o orçamento de subtokens, roda a previsão em cada janela e mescla os resultados usando só a
   faixa "núcleo" de cada janela (fora da sobreposição com a vizinha), pra não fatiar uma
   entidade bem na fronteira nem duplicá-la na região compartilhada.

   O mesmo truncamento acontece no TREINO (notebook 06) -- medição em 2026-09-15 sobre os 999
   documentos reais do gabarito: 70,2% das páginas passam de 512 subtokens e ~27,8% de todas as
   entidades do gabarito (10.817/38.960) nunca geravam gradiente, porque caíam depois do corte.
   `dividir_em_janelas_por_orcamento` é a função de divisão compartilhada entre a mescla de
   previsões (inferência) e a geração de exemplos de treino (notebook 06,
   `construir_exemplos_documento`), pra manter os dois lados com o mesmo critério de corte.

Todas as funções operam sobre a mesma estrutura de "entidade prevista" usada nos notebooks
06/07/09 (dict com `type`, `bbox` [lista de retângulos], `bbox_tokens`, `indices`, `text`).
"""

import re
import unicodedata

CPF_NOME_FUNDIDO = re.compile(r"^([\d.\-]{11,18})-([A-ZÀ-Ü].+)$")
PALAVRA_MAIUSCULA = re.compile(r"^[A-ZÀ-Ü][A-ZÀ-Üa-zà-ü'.-]*$")
PALAVRAS_CONECTORAS_NOME = {"DA", "DE", "DO", "DAS", "DOS", "E"}


def extrair_nome_assinatura_digital(tokens: list, tokens_bbox: list) -> list:
    """Varre os tokens de uma página em busca do padrão "ASSINADO DIGITALMENTE POR:
    <CPF>-<NOME>" e devolve as entidades NOME encontradas (uma lista de dicts no mesmo
    formato usado por `decodificar_bio` nos notebooks 06/07).

    Não emite a entidade CPF correspondente: como CPF e nome saem fundidos no mesmo
    token, o bbox do CPF isolado não é recuperável com precisão suficiente para IoU —
    só o nome (que domina a contagem de erro) é extraído aqui.
    """
    entidades = []
    n = len(tokens)
    i = 0
    while i < n - 2:
        janela = " ".join(tokens[i:i + 3]).upper().rstrip(":")
        if janela.startswith("ASSINADO DIGITALMENTE POR"):
            j = i + 3
            # o token seguinte normalmente é "<CPF>-<PRIMEIRO_NOME>" fundido (às vezes com
            # o "POR:" e o CPF já no mesmo token de índice i+2, daí o rstrip acima)
            candidatos_idx = [j] if j < n else []
            if i + 2 < n and CPF_NOME_FUNDIDO.match(tokens[i + 2]):
                candidatos_idx = [i + 2]
            for idx in candidatos_idx:
                if idx >= n:
                    continue
                m = CPF_NOME_FUNDIDO.match(tokens[idx])
                if not m:
                    continue
                partes_nome = [m.group(2)]
                indices_nome = [idx]
                k = idx + 1
                while k < n and len(partes_nome) < 8:
                    tok = tokens[k]
                    if PALAVRA_MAIUSCULA.match(tok) and (tok.isupper() or tok.upper() in PALAVRAS_CONECTORAS_NOME):
                        partes_nome.append(tok)
                        indices_nome.append(k)
                        k += 1
                    else:
                        break
                texto_nome = " ".join(partes_nome)
                if len(texto_nome.split()) >= 2:  # exige pelo menos 2 palavras pra reduzir falso positivo
                    caixas = [tokens_bbox[t] for t in indices_nome]
                    entidades.append({
                        "type": "NOME",
                        "bbox": [[min(b[0] for b in caixas), min(b[1] for b in caixas),
                                   max(b[2] for b in caixas), max(b[3] for b in caixas)]],
                        "bbox_tokens": caixas,
                        "indices": indices_nome,
                        "text": texto_nome,
                        "origem": "regex_assinatura_digital",
                    })
                break
            i = j
        else:
            i += 1
    return entidades


def estender_boundary_nome(entidades_previstas: list, tokens: list, tokens_bbox: list, max_palavras_total: int = 6) -> list:
    """Estende entidades NOME que pararam no meio de um nome composto, absorvendo tokens
    vizinhos que parecem continuar o mesmo nome (maiúsculas ou conectores DA/DE/DO/DOS/DAS/E),
    até `max_palavras_total` palavras. Não muda entidades de outros tipos.

    Heurística, não uma correção definitiva — documentar qualquer ganho/perda de precisão
    ao usar (ver anotacoes.txt 2026-09-11)."""
    n = len(tokens)
    resultado = []
    indices_usados_por_outras = set()
    for e in entidades_previstas:
        if e["type"] != "NOME":
            resultado.append(e)
            continue
        indices = list(e["indices"])
        ultimo = max(indices)
        palavras_atuais = len(indices)
        k = ultimo + 1
        while k < n and palavras_atuais < max_palavras_total:
            tok = tokens[k]
            if PALAVRA_MAIUSCULA.match(tok) and (tok.isupper() or tok.upper() in PALAVRAS_CONECTORAS_NOME):
                indices.append(k)
                palavras_atuais += 1
                k += 1
            else:
                break
        if indices != e["indices"]:
            caixas = [tokens_bbox[i] for i in indices]
            nova = dict(e)
            nova["indices"] = indices
            nova["text"] = " ".join(tokens[i] for i in indices)
            nova["bbox"] = [[min(b[0] for b in caixas), min(b[1] for b in caixas),
                               max(b[2] for b in caixas), max(b[3] for b in caixas)]]
            nova["bbox_tokens"] = caixas
            nova["origem"] = e.get("origem", "modelo") + "+boundary_estendido"
            resultado.append(nova)
        else:
            resultado.append(e)
    return resultado


_PONTUACAO_BORDA = " \t\n\r.,;:()[]{}\"'`´“”‘’–—-"


def _normalizar_token(tok: str) -> str:
    """Maiúsculas, sem pontuação de borda, sem acento -- para comparação de texto na
    propagação. Sem a remoção de acento, "SERGIO" (ex.: bloco de assinatura digital, sem
    acentuação) e "SÉRGIO" (ex.: parágrafo de qualificação do sócio, com acentuação) nunca
    batem como o mesmo texto, e a propagação nunca alcança a variante acentuada (achado
    2026-09-15, caso real "SÉRGIO LÚCIO SOARES" no holdout)."""
    tok = tok.strip(_PONTUACAO_BORDA).upper()
    tok = unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode("ascii")
    return tok


def propagar_entidades_por_texto_identico(
    entidades_documento: list,
    paginas: list,
    tipos_propagaveis: tuple = ("NOME",),
    min_palavras_por_tipo: dict = None,
) -> list:
    """Repropaga entidades já detectadas para outras ocorrências do mesmo texto exato
    dentro do documento, em páginas/posições onde o modelo não marcou nada.

    `entidades_documento`: lista de entidades já previstas (todas as páginas, mesmo formato
    usado nos notebooks -- precisa de `type`, `text`, `indices`, `page`).
    `paginas`: lista de dicts no formato do JSON de extração, cada um com `page` e `tokens`
    (lista de dicts com `text`/`bbox`) -- o mesmo `extracao["pages"]` usado para gerar as
    previsões originais.
    `tipos_propagaveis`: quais tipos podem ser propagados (default só NOME, que é onde o
    problema foi observado -- CPF/RG/ENDERECO podem ser habilitados também, ver notebook 09).
    `min_palavras_por_tipo`: mínimo de palavras para um texto virar "canônico" e ser
    propagado, por tipo -- default `{"NOME": 2}` (mesmo critério de
    `extrair_nome_assinatura_digital`, reduz o risco de propagar um sobrenome comum sozinho
    como falso positivo). Tipos fora do dict usam mínimo 1 -- importante para CPF/RG, que
    normalmente saem como um único token; um mínimo de 2 palavras nunca bateria com eles e
    a propagação para esses tipos ficaria silenciosamente sem efeito.

    Quando o span canônico encontrado numa página colide com uma entidade já existente ali:
      - se a entidade existente já cobre o span completo (ou mais), não mexe em nada;
      - se a entidade existente cobre só uma PARTE do span (ex.: modelo previu "302" onde o
        texto completo do carimbo é "AV. BIAS FORTES, 932 CONJ. 302 [...]"), ela é
        DESCARTADA e substituída pela entidade de span completo -- prioriza consistência de
        marcação entre páginas idênticas sobre a robustez de cada previsão isolada (decisão
        de produto 2026-09-15, ver módulo).

    ATENÇÃO -- mudança de contrato: devolve a lista COMPLETA e já corrigida de entidades do
    documento (não só as novas). Chame substituindo a lista antiga, não concatenando:
        entidades_documento = propagar_entidades_por_texto_identico(entidades_documento, ...)
    (usar `entidades_documento = entidades_documento + propagar_...(...)` ou `.extend(...)`
    duplica entidades, porque o retorno já inclui as que não foram descartadas.)
    Cada entidade nova ganha `origem="propagado_texto_identico"` e a `page` correspondente.
    """
    if min_palavras_por_tipo is None:
        min_palavras_por_tipo = {"NOME": 2}

    tokens_por_pagina = {
        p["page"]: ([t["text"] for t in p["tokens"]], [t["bbox"] for t in p["tokens"]])
        for p in paginas
        if p["tokens"]
    }

    canonicos: dict = {}
    for e in entidades_documento:
        if e["type"] not in tipos_propagaveis:
            continue
        palavras = [_normalizar_token(p) for p in e["text"].split()]
        # remove pontuação solta só das PONTAS (ex.: vírgula/travessão que ficou como token
        # separado bem no início/fim do span, comum quando a fronteira da entidade "vaza" um
        # pouco) -- um separador isolado no MEIO do texto (ex.: travessão entre partes de um
        # endereço: "302 - LOURDES - BELO HORIZONTE/MG") continua fazendo parte da chave.
        # Sem esse trim, a MESMA entidade com/sem uma vírgula anexada na borda virava dois
        # canônicos distintos (achado 2026-09-15, segunda rodada) e o mais comprido acabava
        # descartando o mais curto (e vice-versa) mesmo quando os dois já estavam corretos.
        while palavras and not palavras[0]:
            palavras.pop(0)
        while palavras and not palavras[-1]:
            palavras.pop()
        if len(palavras) < min_palavras_por_tipo.get(e["type"], 1):
            continue
        chave = tuple(palavras)
        if not chave or chave in canonicos:
            continue
        canonicos[chave] = e["type"]

    entidades_por_pagina: dict = {}
    for e in entidades_documento:
        entidades_por_pagina.setdefault(e["page"], []).append(e)

    resultado = []
    paginas_processadas = set()

    for pagina, (tokens, tokens_bbox) in tokens_por_pagina.items():
        paginas_processadas.add(pagina)
        normalizados = [_normalizar_token(t) for t in tokens]
        n_tokens = len(tokens)
        existentes = entidades_por_pagina.get(pagina, [])

        cobertura = {id(e): set(e["indices"]) for e in existentes}
        descartadas_ids = set()
        usados = set()
        for e in existentes:
            usados.update(cobertura[id(e)])

        novas_da_pagina = []
        for chave, tipo in canonicos.items():
            n = len(chave)
            i = 0
            while i <= n_tokens - n:
                if tuple(normalizados[i:i + n]) != chave:
                    i += 1
                    continue

                indices_match = set(range(i, i + n))
                colidentes = [
                    e for e in existentes
                    if id(e) not in descartadas_ids and cobertura[id(e)] & indices_match
                ]

                if colidentes:
                    if any(indices_match <= cobertura[id(e)] for e in colidentes):
                        # alguma entidade já existente cobre o span inteiro -- não mexe
                        i += n
                        continue
                    # todas as colisões são span PARCIAL -- descarta e substitui pelo completo
                    for e in colidentes:
                        descartadas_ids.add(id(e))
                        usados -= cobertura[id(e)]
                elif indices_match & usados:
                    # posição já preenchida por outra propagação nesta mesma passada
                    i += n
                    continue

                indices_lista = sorted(indices_match)
                caixas = [tokens_bbox[k] for k in indices_lista]
                novas_da_pagina.append({
                    "type": tipo,
                    "bbox": [[min(b[0] for b in caixas), min(b[1] for b in caixas),
                               max(b[2] for b in caixas), max(b[3] for b in caixas)]],
                    "bbox_tokens": caixas,
                    "indices": indices_lista,
                    "text": " ".join(tokens[k] for k in indices_lista),
                    "page": pagina,
                    "origem": "propagado_texto_identico",
                })
                usados.update(indices_match)
                i += n

        for e in existentes:
            if id(e) not in descartadas_ids:
                resultado.append(e)
        resultado.extend(novas_da_pagina)

    # páginas sem token algum (raro) -- preserva as entidades como estavam, sem tentar propagar
    for pagina, ents in entidades_por_pagina.items():
        if pagina not in paginas_processadas:
            resultado.extend(ents)

    return resultado


def comprimentos_subtokens(tokenizer, tokens: list) -> list:
    """Nº de subtokens (wordpieces) que cada palavra de `tokens` vira depois da tokenização
    do BERT -- base para decidir onde cortar as janelas (usado tanto no treino quanto na
    inferência, ver `dividir_em_janelas_por_orcamento`/`mesclar_previsoes_por_janela`)."""
    return [len(tokenizer.tokenize(t)) or 1 for t in tokens]


def dividir_em_janelas_por_orcamento(comprimentos: list, orcamento: int, sobreposicao_palavras: int) -> list:
    """Divide uma sequência de `comprimentos` (subtokens por palavra) em janelas
    `[inicio, fim)` (índices de palavra) que cabem no `orcamento` de subtokens, com
    `sobreposicao_palavras` de sobreposição entre janelas vizinhas -- pra não fatiar uma
    entidade bem na fronteira. Usado tanto para mesclar previsões na inferência
    (`mesclar_previsoes_por_janela`) quanto para gerar exemplos de treino a partir de
    páginas longas (ver notebook 06, achado 2026-09-15: ~28% das entidades do gabarito real
    nunca eram vistas no treino por causa do truncamento silencioso em MAX_LENGTH)."""
    n = len(comprimentos)
    if n == 0:
        return [(0, 0)]
    janelas = []
    inicio = 0
    while inicio < n:
        soma, fim = 0, inicio
        while fim < n and soma + comprimentos[fim] <= orcamento:
            soma += comprimentos[fim]
            fim += 1
        if fim == inicio:  # uma palavra isolada já estoura o orçamento -- inclui mesmo assim
            fim = inicio + 1
        janelas.append((inicio, fim))
        if fim >= n:
            break
        inicio = max(fim - sobreposicao_palavras, inicio + 1)
    return janelas


def mesclar_previsoes_por_janela(
    tokens: list,
    tokenizer,
    prever_tags_janela: callable,
    max_length: int = 512,
    sobreposicao_palavras: int = 64,
) -> list:
    """Roda `prever_tags_janela` em janelas deslizantes quando a página, tokenizada em
    subtokens, ultrapassa `max_length` -- ver item 4 no docstring do módulo (achado
    2026-09-15, caso MINAS CAL/endereço do segundo sócio). Quando a página cabe inteira no
    orçamento de subtokens, roda uma única vez, sem overhead (comportamento idêntico ao
    código anterior a essa correção).

    `prever_tags_janela(tokens_da_janela) -> list[str]`: callback que devolve uma tag BIO
    ("O"/"B-TIPO"/"I-TIPO") por palavra da janela recebida -- mesmo contrato/mesma função
    já usada para uma página que cabe inteira, só que chamada uma vez por janela em vez de
    uma vez pela página toda. Funciona tanto para o modelo de cabeça única (chame uma vez
    por página) quanto para o de 4 cabeças (chame uma vez por página POR TIPO, já que cada
    cabeça tem sua própria sequência de tags).

    Retorna a lista de tags cobrindo TODAS as `len(tokens)` palavras da página (nunca corta
    silenciosamente) -- mesmo formato de retorno de `prever_tags_janela` para o caso de
    janela única, então o chamador não precisa saber se houve chunking ou não.
    """
    n = len(tokens)
    if n == 0:
        return []

    comprimentos = comprimentos_subtokens(tokenizer, tokens)
    orcamento = max_length - 2  # reserva para [CLS]/[SEP]
    if sum(comprimentos) <= orcamento:
        return prever_tags_janela(tokens)

    janelas = dividir_em_janelas_por_orcamento(comprimentos, orcamento, sobreposicao_palavras)

    tags_globais = ["O"] * n
    for idx_janela, (ini, fim) in enumerate(janelas):
        tags_janela = prever_tags_janela(tokens[ini:fim])
        # faixa "núcleo": só aceita previsões desta janela fora da sobreposição com a
        # vizinha, pra não fatiar uma entidade bem na fronteira nem duplicá-la na região
        # compartilhada por duas janelas.
        nucleo_ini = ini if idx_janela == 0 else ini + sobreposicao_palavras // 2
        nucleo_fim = fim if idx_janela == len(janelas) - 1 else fim - sobreposicao_palavras // 2
        nucleo_fim = max(nucleo_fim, nucleo_ini + 1)
        for i_global in range(max(ini, nucleo_ini), min(fim, nucleo_fim)):
            tags_globais[i_global] = tags_janela[i_global - ini]

    return tags_globais
