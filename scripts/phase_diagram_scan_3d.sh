#!/bin/bash
# Generate U-T phase diagram data for the attractive Hubbard model
# using the 3D Block RG example with superconducting correlation diagnostics.
#
# Usage: ./scripts/phase_diagram_scan_3d.sh [output_dir]
#
# Outputs:
#   - phase_diagram_3d_data.dat: Raw data (U, T, lambda_pair, pair_corr)
#   - phase_diagram_3d_combined.png: Side-by-side heatmaps
#   - phase_diagram_3d_lambda_pair.png: Pairing amplitude heatmap
#   - phase_diagram_3d_pair_corr.png: Pair correlation heatmap

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$PROJECT_DIR}"

BRG_EXECUTABLE="$PROJECT_DIR/build/examples/example_hubbard_block_rg_3d_finite_t"

if [[ ! -x "$BRG_EXECUTABLE" ]]; then
    echo "Error: BRG executable not found at $BRG_EXECUTABLE"
    echo "Please build the project first: ./scripts/make.sh"
    exit 1
fi

echo "============================================================="
echo "  Attractive Hubbard Model: U-T Phase Diagram Scan (3D)"
echo "  Using 2x2x2 Block RG at 1/8 filling (Mode B)"
echo "============================================================="
echo ""

# Grid parameters
U_VALUES=($(seq 0.0 -1.0 -10.0))
T_VALUES=($(seq 0.0 0.1 1.0))

DATA_FILE="$OUTPUT_DIR/phase_diagram_3d_data.dat"

echo "Generating ${#U_VALUES[@]}x${#T_VALUES[@]} grid data..."
echo "Output: $DATA_FILE"
echo ""

# Create data file with header
cat > "$DATA_FILE" << 'EOF'
# U-T Phase Diagram for Attractive Hubbard Model (2x2x2 Block RG, 3D)
# Columns: U  T  lambda_pair  pair_corr
# lambda_pair = <psi_N2|c^dag_up c^dag_down|psi_N0> (pairing amplitude)
# pair_corr = <Delta^dag_i Delta_j> for i!=j (inter-site pair correlation)
EOF

total=$((${#U_VALUES[@]} * ${#T_VALUES[@]}))
count=0

for U in "${U_VALUES[@]}"; do
    for T in "${T_VALUES[@]}"; do
        result=$("$BRG_EXECUTABLE" -U "$U" --mode B -n 1 --temperature "$T" 2>/dev/null | grep -v "^#")
        lambda_pair=$(echo "$result" | awk '{print $11}')
        pair_corr=$(echo "$result" | awk '{print $12}')
        echo "$U $T $lambda_pair $pair_corr" >> "$DATA_FILE"

        count=$((count + 1))
        printf "\r  Progress: %d/%d (U=%.1f, T=%.1f)" "$count" "$total" "$U" "$T"
    done
    echo "" >> "$DATA_FILE"  # Blank line for gnuplot pm3d
done

echo ""
echo "Data generation complete."
echo ""

# Check if gnuplot is available
if ! command -v gnuplot &> /dev/null; then
    echo "Warning: gnuplot not found. Skipping plot generation."
    echo "Data file created: $DATA_FILE"
    exit 0
fi

echo "Generating plots with gnuplot..."

# Plot 1: lambda_pair heatmap
cat > /tmp/plot_3d_lambda_pair.gp << EOF
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output '$OUTPUT_DIR/phase_diagram_3d_lambda_pair.png'

set title "Pairing Amplitude λ_{pair} = ⟨ψ_{N=2}|c^†_↑c^†_↓|ψ_{N=0}⟩" font 'Arial,14'
set xlabel "|U|/t (Interaction Strength)" font 'Arial,12'
set ylabel "T/t (Temperature)" font 'Arial,12'

set pm3d map
set pm3d interpolate 4,4

set palette defined (0.25 "dark-blue", 0.30 "blue", 0.35 "cyan", 0.40 "green", 0.45 "yellow", 0.50 "red")
set cbrange [0:*]
set cblabel "λ_{pair}" font 'Arial,12'

splot '$DATA_FILE' using 1:2:3 with pm3d notitle
EOF
gnuplot /tmp/plot_3d_lambda_pair.gp
echo "  Created: $OUTPUT_DIR/phase_diagram_3d_lambda_pair.png"

# Plot 2: pair_corr heatmap
cat > /tmp/plot_3d_pair_corr.gp << EOF
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output '$OUTPUT_DIR/phase_diagram_3d_pair_corr.png'

set title "Inter-site Pair Correlation ⟨Δ^†_i Δ_j⟩ (i≠j)" font 'Arial,14'
set xlabel "|U|/t (Interaction Strength)" font 'Arial,12'
set ylabel "T/t (Temperature)" font 'Arial,12'

set pm3d map
set pm3d interpolate 4,4

set palette defined (0 "white", 0.02 "dark-blue", 0.05 "blue", 0.10 "cyan", 0.15 "green", 0.20 "yellow", 0.23 "orange", 0.25 "red")
set cbrange [0:*]
set cblabel "⟨Δ^†_i Δ_j⟩" font 'Arial,12'

splot '$DATA_FILE' using 1:2:4 with pm3d notitle
EOF
gnuplot /tmp/plot_3d_pair_corr.gp
echo "  Created: $OUTPUT_DIR/phase_diagram_3d_pair_corr.png"

# Plot 3: Combined side-by-side
cat > /tmp/plot_3d_combined.gp << EOF
set terminal pngcairo size 1400,600 enhanced font 'Arial,12'
set output '$OUTPUT_DIR/phase_diagram_3d_combined.png'

set multiplot layout 1,2 title "Attractive Hubbard Model: Superconducting Correlations (2x2x2 Block RG, 3D)" font 'Arial,16'

# Left plot: lambda_pair
set title "Pairing Amplitude λ_{pair}" font 'Arial,14'
set xlabel "|U|/t" font 'Arial,12'
set ylabel "T/t" font 'Arial,12'

set pm3d map
set pm3d interpolate 4,4

set palette defined (0.25 "dark-blue", 0.30 "blue", 0.35 "cyan", 0.40 "green", 0.45 "yellow", 0.50 "red")
set cbrange [0:*]
set cblabel "λ_{pair}" font 'Arial,11'

splot '$DATA_FILE' using 1:2:3 with pm3d notitle

# Right plot: pair_corr
set title "Inter-site Pair Correlation ⟨Δ^†_i Δ_j⟩" font 'Arial,14'
set xlabel "|U|/t" font 'Arial,12'
set ylabel "T/t" font 'Arial,12'

set palette defined (0 "white", 0.02 "dark-blue", 0.05 "blue", 0.10 "cyan", 0.15 "green", 0.20 "yellow", 0.23 "orange", 0.25 "red")
set cbrange [0:*]
set cblabel "⟨Δ^†_i Δ_j⟩" font 'Arial,11'

splot '$DATA_FILE' using 1:2:4 with pm3d notitle

unset multiplot
EOF
gnuplot /tmp/plot_3d_combined.gp
echo "  Created: $OUTPUT_DIR/phase_diagram_3d_combined.png"

# Cleanup
rm -f /tmp/plot_3d_lambda_pair.gp /tmp/plot_3d_pair_corr.gp /tmp/plot_3d_combined.gp

echo ""
echo "============================================================="
echo "  Phase diagram generation complete!"
echo "============================================================="
echo ""
echo "Output files:"
echo "  $OUTPUT_DIR/phase_diagram_3d_data.dat"
echo "  $OUTPUT_DIR/phase_diagram_3d_lambda_pair.png"
echo "  $OUTPUT_DIR/phase_diagram_3d_pair_corr.png"
echo "  $OUTPUT_DIR/phase_diagram_3d_combined.png"
echo ""
echo "Physical interpretation:"
echo "  - lambda_pair: Local pairing amplitude (0.25=non-interacting, 0.5=maximal)"
echo "  - pair_corr: Inter-site pair coherence (order parameter for SC)"
echo "  - Strong SC correlations at low T, large |U| (bottom-left of plots)"
echo "  - Temperature suppresses pairing correlations"
