#!/usr/bin/env bash
# setup_tera_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════
# Cria a árvore completa de diretórios do TERA Pipeline e posiciona
# os arquivos nos locais corretos.
#
# USO:
#   chmod +x setup_tera_pipeline.sh
#   ./setup_tera_pipeline.sh
#
# PRÉ-REQUISITO:
#   Execute na raiz do projeto, onde já existem:
#   - run_v29_ablacao_ttdef_ajuste_gatting.py
#   - synthetic_driver_risk_v7.py
#   - modelos_salvos/abl_A/
#   - dados_sinteticos/
# ═══════════════════════════════════════════════════════════════════════════

set -e
echo "═══════════════════════════════════════════════════════════"
echo "  TERA Pipeline v1.0 — Setup de Diretórios"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Criar árvore de diretórios ────────────────────────────────────────────
echo ""
echo "[1/4] Criando árvore de diretórios..."

mkdir -p tera_pipeline/{dataset,training,calibration,inference,evaluation,logging,figures,export,latex,utils}
mkdir -p configs
mkdir -p results
mkdir -p _legacy
mkdir -p paper

echo "  ✓ Árvore criada"

# ── 2. Criar __init__.py em todos os submódulos ──────────────────────────────
echo ""
echo "[2/4] Criando __init__.py..."

cat > tera_pipeline/__init__.py << 'EOF'
"""TERA Pipeline v1.0 — Framework Experimental Reproduzível para FGCS/Elsevier"""
__version__ = "1.0.0"
from tera_pipeline.utils.tera_utils import set_global_seed, get_device, load_config, setup_logging
EOF

for submod in dataset training calibration inference evaluation logging figures export latex utils; do
    touch "tera_pipeline/${submod}/__init__.py"
done
echo "  ✓ __init__.py criados"

# ── 3. Posicionar arquivos dos módulos ────────────────────────────────────────
echo ""
echo "[3/4] Posicionando módulos..."

# Diretório onde os módulos foram baixados (ajuste se necessário)
MODULES_DIR="${1:-.}"

MODULE_MAP=(
    "tera_gen.py:tera_pipeline/dataset/tera_gen.py"
    "tera_train.py:tera_pipeline/training/tera_train.py"
    "tera_calibrate.py:tera_pipeline/calibration/tera_calibrate.py"
    "tera_infer.py:tera_pipeline/inference/tera_infer.py"
    "tera_eval.py:tera_pipeline/evaluation/tera_eval.py"
    "tera_log.py:tera_pipeline/logging/tera_log.py"
    "tera_figures.py:tera_pipeline/figures/tera_figures.py"
    "tera_export.py:tera_pipeline/export/tera_export.py"
    "tera_latex.py:tera_pipeline/latex/tera_latex.py"
    "tera_utils.py:tera_pipeline/utils/tera_utils.py"
    "experiment_config.yaml:configs/experiment_config.yaml"
    "run_experiment.py:run_experiment.py"
    "ARCHITECTURE.md:ARCHITECTURE.md"
    "MIGRATION_GUIDE.md:MIGRATION_GUIDE.md"
)

for entry in "${MODULE_MAP[@]}"; do
    src="${MODULES_DIR}/${entry%%:*}"
    dst="${entry##*:}"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "  ✓ $dst"
    else
        echo "  ⚠  Não encontrado: $src (copie manualmente)"
    fi
done

# ── 4. Mover scripts legados ─────────────────────────────────────────────────
echo ""
echo "[4/4] Arquivando scripts legados em _legacy/..."

LEGACY=(
    "patch_baseline_hibrida.py"
    "patch_ttdef_estratificado.py"
    "patch_v29_ttd_paper.py"
    "patch_v29_ttd_paper_corrigido.py"
    "run_hybrid_completion.py"
    "run_inferencia_seed42_corrigido.py"
    "script.py"
    "script_2.py"
    "script_3.py"
    "script_ver_final.py"
    "fracos.py"
)

for f in "${LEGACY[@]}"; do
    if [ -f "$f" ]; then
        mv "$f" "_legacy/$f"
        echo "  → _legacy/$f"
    fi
done

if [ -f "run_v29_ablacao_ttdef_ajuste_gatting.py" ]; then
    cp "run_v29_ablacao_ttdef_ajuste_gatting.py" "_legacy/run_v29_legacy.py"
    echo "  → _legacy/run_v29_legacy.py (cópia de referência)"
fi

# ── Resumo ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup concluído!"
echo ""
echo "  Próximos passos:"
echo "  1. Verificar configuração:"
echo "     python run_experiment.py --dry-run"
echo ""
echo "  2. Executar pipeline (reutilizando modelos existentes):"
echo "     python run_experiment.py --reuse-data --reuse-models"
echo ""
echo "  3. Gerar apenas macros (modelos e dados já existem):"
echo "     python run_experiment.py --reuse-data --reuse-models \\"
echo "            --stages calibration,inference,evaluation,export"
echo "═══════════════════════════════════════════════════════════"
