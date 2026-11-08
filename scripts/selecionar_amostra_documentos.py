r"""
Seleciona uma amostra ALEATÓRIA (não sequencial, para não pegar dados enviesados pela ordem
de inserção no banco de origem) de documentos da pasta de origem e copia para
data/documento_social/01.arquivos_originais/.

Uso:
    python scripts\selecionar_amostra_documentos.py --origem "D:\Arquivos\ProjetosPython\BFC\LocalScanDocument_v2\data\ocix\doc_social\files" --n 500

Seed fixa (42) para a amostra ser reprodutível — documentar a seed usada no anotacoes.txt
quando rodar, para fins de metodologia do artigo.

Nunca reseleciona arquivos que já estão em `01.arquivos_originais/arquivos_reais/` (treino/
validação, ver notebooks 01/03/06) NEM em `teste_holdout/arquivos/` (conjunto de teste
holdout, ver `selecionar_amostra_teste_holdout.py` e notebooks 08/09) — evita que o mesmo
documento apareça tanto no treino/validação quanto no teste holdout.
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
    parser.add_argument("--n", type=int, default=500, help="Quantos documentos selecionar (padrão: 500)")
    parser.add_argument(
        "--destino",
        default=None,
        help="Pasta de destino (padrão: data/documento_social/01.arquivos_originais/arquivos_reais/ na raiz do projeto)",
    )
    args = parser.parse_args()

    pasta_origem = Path(args.origem)
    if not pasta_origem.is_dir():
        raise SystemExit(f"Pasta de origem não encontrada: {pasta_origem}")

    raiz = Path(__file__).resolve().parent.parent
    if args.destino:
        pasta_destino = Path(args.destino)
    else:
        pasta_destino = raiz / "data" / "documento_social" / "01.arquivos_originais" / "arquivos_reais"
    pasta_destino.mkdir(parents=True, exist_ok=True)

    pasta_teste_holdout = raiz / "data" / "documento_social" / "teste_holdout" / "arquivos"

    ja_existentes = {p.name for p in pasta_destino.iterdir() if p.is_file()}
    ja_no_holdout = {p.name for p in pasta_teste_holdout.iterdir() if p.is_file()} if pasta_teste_holdout.is_dir() else set()
    n_ja_existentes = len(ja_existentes)
    print(f"Já existem {n_ja_existentes} arquivo(s) em {pasta_destino} (não serão reselecionados)")
    print(f"Já existem {len(ja_no_holdout)} arquivo(s) em {pasta_teste_holdout} (também excluídos da seleção)")
    nomes_excluidos = ja_existentes | ja_no_holdout

    candidatos = sorted(
        p for p in pasta_origem.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES_SUPORTADAS and p.name not in nomes_excluidos
    )
    print(f"Candidatos disponíveis na origem (excluindo já existentes): {len(candidatos)}")

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
    print(f"Total agora em {pasta_destino}: {n_ja_existentes + copiados} arquivo(s)")
    print(f"\nSeed usada: {SEED} (registrar no anotacoes.txt para reprodutibilidade)")


if __name__ == "__main__":
    main()
