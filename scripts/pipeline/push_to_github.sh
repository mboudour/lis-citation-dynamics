#!/bin/bash
# Auto-push completed results to GitHub after each stage
# Usage: bash push_to_github.sh [stage_label]

STAGE="${1:-update}"
REPO_DIR="/home/ubuntu/lis_repo"
LIS_DIR="/home/ubuntu/lis"

echo "=== Pushing to GitHub: $STAGE ==="

# Ensure repo dir exists and is a git repo
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "ERROR: $REPO_DIR is not a git repo"
    exit 1
fi

# Copy scripts
cp $LIS_DIR/oa_stage1_features.py $REPO_DIR/code/ 2>/dev/null
cp $LIS_DIR/oa_stage2_sbert.py $REPO_DIR/code/ 2>/dev/null
cp $LIS_DIR/oa_stage2_compute_sims.py $REPO_DIR/code/ 2>/dev/null
cp $LIS_DIR/oa_stage3_ml.py $REPO_DIR/code/ 2>/dev/null
cp $LIS_DIR/oa_stage3_ml_nosbert.py $REPO_DIR/code/ 2>/dev/null
cp $LIS_DIR/oa_stage4_figures.py $REPO_DIR/code/ 2>/dev/null

# Copy results (JSON files only - no large binary files)
mkdir -p $REPO_DIR/results/oa
cp $LIS_DIR/results_oa/*.json $REPO_DIR/results/oa/ 2>/dev/null
cp $LIS_DIR/results_oa/*.csv $REPO_DIR/results/oa/ 2>/dev/null

# Copy figures
mkdir -p $REPO_DIR/figures/oa
cp $LIS_DIR/figures_oa/*.png $REPO_DIR/figures/oa/ 2>/dev/null
cp $LIS_DIR/figures_oa/*.pdf $REPO_DIR/figures/oa/ 2>/dev/null

# Copy manuscript
cp $LIS_DIR/lis_manuscript.tex $REPO_DIR/ 2>/dev/null
cp $LIS_DIR/lis_manuscript.md $REPO_DIR/ 2>/dev/null
cp $LIS_DIR/lis_manuscript.pdf $REPO_DIR/ 2>/dev/null

# Git add, commit, push
cd $REPO_DIR
git add -A
git commit -m "Auto-push: $STAGE [$(date '+%Y-%m-%d %H:%M')]" 2>&1
git push origin master 2>&1
echo "=== Push complete ==="
