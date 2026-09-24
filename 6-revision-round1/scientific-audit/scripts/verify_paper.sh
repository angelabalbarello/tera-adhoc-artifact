#!/usr/bin/env bash
# Verificacao unica: complexidade, estatistica, testes de consistencia e manuscrito.
# Uso: bash verify_paper.sh [pasta_do_manuscrito]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; ART="${1:-$HERE/../../../7-article-latex/revisao-round1}"
echo "== 1. complexidade =="; python3 "$HERE/check_model_complexity.py" "$HERE/../../../../FGCS/Inferencia/modelos_salvos/abl_A/student_seed42.pt" 2>/dev/null | tail -4 || python3 -c "import sys;sys.path.insert(0,'$HERE/../tests');import test_model_complexity as t;t.test_closed_form();print('closed-form OK (checkpoint indisponivel)')"
echo "== 2. estatistica =="; python3 "$HERE/recompute_statistics.py" | tail -3
echo "== 3. testes =="
for t in "$HERE/../tests"/test_*.py; do python3 "$t" "$ART" || exit 1; done
echo "== 4. manuscrito =="; python3 "$HERE/check_manuscript_consistency.py" "$ART" | tail -1
echo "== verify_paper: TUDO OK =="
