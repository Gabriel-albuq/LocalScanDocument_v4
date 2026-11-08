r"""
Seleciona uma amostra ALEATÓRIA de documentos da pasta de origem (o mesmo banco de onde vêm
os documentos reais, ver `selecionar_amostra_documentos.py`) e copia para
data/documento_social/teste_holdout/arquivos/ — o conjunto de teste HOLDOUT usado pelos
notebooks 08/09 para testar o modelo em documentos genuinamente nunca vistos.

Nunca seleciona um documento que já esteja em `01.arquivos_originais/arquivos_reais/`
(treino/validação, ver notebooks 01/03/06) nem um que já esteja em `teste_holdout/arquivos/`
(rodadas anteriores deste mesmo script) — garante que treino/validação e teste holdout nunca
se sobrepõem.

Uso:
    python scripts\selecionar_amostra_teste_holdout.py --origem "D:\Arquivos\ProjetosPython\BFC\LocalScanDocument_v2\data\ocix\doc_social\files" --n 30

Seed fixa (42) para a amostra ser reprodutível — documentar a seed usada no anotacoes.txt
quando rodar, para fins de metodologia do artigo. Como a exclusão de `arquivos_reais` e do
próprio holdout muda o conjunto de candidatos a cada rodada, a MESMA seed não repete a mesma
amostra se o conteúdo dessas pastas mudar entre execuções — isso é esperado (a garantia é
"nunca escolhe um documento já usado em outro conjunto", não "a mesma amostra sempre").
"""

import argparse
import random
import shutil
from pathlib import Path

SEED = 42
EXTENSOES_SUPORTADAS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origem", required=True, help="Pasta de origem com os documentos")
    parser.add_argument("--n", type=int, default=30, help="Quantos documentos selecionar (padrão: 30)")
    parser.add_argument(
        "--destino",
        default=None,
        help="Pasta de destino (padrão: data/documento_social/teste_holdout/arquivos/ na raiz do projeto)",
    )
    args = parser.parse_args()

    pasta_origem = Path(args.origem)
    if not pasta_origem.is_dir():
        raise SystemExit(f"Pasta de origem não encontrada: {pasta_origem}")

    raiz = Path(__file__).resolve().parent.parent
    if args.destino:
        pasta_destino = Path(args.destino)
    else:
        pasta_destino = raiz / "data" / "documento_social" / "teste_holdout" / "arquivos"
    pasta_destino.mkdir(parents=True, exist_ok=True)

    pasta_treino_validacao = raiz / "data" / "documento_social" / "01.arquivos_originais" / "arquivos_reais"

    ja_no_holdout = {p.name for p in pasta_destino.iterdir() if p.is_file()}
    ja_treino_validacao = (
        {p.name for p in pasta_treino_validacao.iterdir() if p.is_file()} if pasta_treino_validacao.is_dir() else set()
    )
    print(f"Já existem {len(ja_no_holdout)} arquivo(s) em {pasta_destino} (não serão reselecionados)")
    print(f"Existem {len(ja_treino_validacao)} arquivo(s) em {pasta_treino_validacao} (excluídos da seleção -- treino/validação)")
    nomes_excluidos = ja_no_holdout | ja_treino_validacao

    candidatos = sorted(
        p for p in pasta_origem.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES_SUPORTADAS and p.name not in nomes_excluidos
    )
    print(f"Candidatos disponíveis na origem (nunca vistos em treino/validação/holdout): {len(candidatos)}")

    if len(candidatos) < args.n:
        raise SystemExit(f"Só {len(candidatos)} candidatos disponíveis, menos que os {args.n} pedidos.")

    rng = random.Random(SEED)
    selecionados = rng.sample(candidatos, args.n)

    print(f"\nCopiando {len(selecionados)} arquivo(s) (seed={SEED}) para {pasta_destino}...")
    copiados = 0
    for i, caminho in enumerate(selecionados, start=1):
        destino = pasta_destino / caminho.name
        shutil.copyfile(caminho, destino)
        copiados += 1
        if i % 50 == 0 or i == len(selecionados):
            print(f"  {i}/{len(selecionados)} copiados...")

    print(f"\nConcluído: {copiados} arquivo(s) novo(s) copiado(s) para {pasta_destino}")
    print(f"Total agora em {pasta_destino}: {len(ja_no_holdout) + copiados} arquivo(s)")
    print(f"\nSeed usada: {SEED} (registrar no anotacoes.txt para reprodutibilidade)")
    print("Próximo passo: rodar notebooks/07.gerar_ground_truth_holdout.ipynb e "
          "notebooks/08.extrair_texto_holdout.ipynb (kernel locscan-ocr) -- em qualquer ordem -- "
          "depois notebooks/09.alinhar_ground_truth_holdout.ipynb, "
          "notebooks/10.extrair_metricas_holdout.ipynb e "
          "notebooks/11.verificar_resultados_manualmente.ipynb (kernel geral).")


if __name__ == "__main__":
    main()
