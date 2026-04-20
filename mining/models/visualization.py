"""
Mining Simulation Visualization Module
=======================================

Publication-quality figures for uranium mining simulation results.
Generates individual PNGs and a combined multi-panel PDF suitable
for academic papers and presentations.

Requires: matplotlib, numpy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


# Academic style defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})


class MiningVisualization:
    """Generate publication-quality figures from mining simulation results."""

    def __init__(self, results: Dict, output_dir: str = "figures"):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures = []

    def generate_all(self) -> str:
        """Generate all figures and a combined PDF report."""
        print("Generating publication-quality figures...")

        if 'resource_estimation' in self.results:
            self._plot_grade_distribution()
            self._plot_grade_tonnage_curve()
            self._plot_resource_classification()

        if 'extraction_efficiency' in self.results:
            self._plot_extraction_recovery()
            self._plot_concentration_profiles()

        if 'environmental_impact' in self.results:
            self._plot_dose_pathways()
            self._plot_radionuclide_transport()

        if 'mine_planning' in self.results:
            self._plot_economic_evaluation()
            self._plot_production_schedule()

        if 'uncertainty' in self.results:
            self._plot_monte_carlo_histogram()
            self._plot_sensitivity_tornado()

        # Combined PDF
        pdf_path = self._save_combined_pdf()
        plt.close('all')

        print(f"  {len(self.figures)} figures saved to: {self.output_dir}/")
        print(f"  Combined PDF: {pdf_path}")
        return str(pdf_path)

    # ── Resource Estimation ─────────────────────────────────────────────

    def _plot_grade_distribution(self):
        """Histogram of estimated ore grades with statistics."""
        re = self.results['resource_estimation']
        grades = np.array(re['estimation_results']['estimated_grades'])
        grades = grades[grades > 0]  # Remove zeros (outside domain)

        fig, ax = plt.subplots(figsize=(7, 5))
        n, bins, patches = ax.hist(grades * 100, bins=30, color='#2196F3',
                                   edgecolor='white', alpha=0.85, density=True)

        # Statistics overlay
        mean_g = np.mean(grades) * 100
        median_g = np.median(grades) * 100
        ax.axvline(mean_g, color='#F44336', linestyle='--', linewidth=1.5,
                    label=f'Mean: {mean_g:.3f}%')
        ax.axvline(median_g, color='#FF9800', linestyle='-.', linewidth=1.5,
                    label=f'Median: {median_g:.3f}%')

        cutoff = self.results['resource_estimation']['parameters'].get('grade_cutoff', 0.05) * 100
        ax.axvline(cutoff, color='#4CAF50', linestyle=':', linewidth=2,
                    label=f'Cutoff: {cutoff:.1f}%')

        ax.set_xlabel('Ore Grade (% U₃O₈)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Estimated Ore Grade Distribution')
        ax.legend(framealpha=0.9)
        self._save_fig(fig, 'grade_distribution')

    def _plot_grade_tonnage_curve(self):
        """Grade-tonnage curve showing resource at different cutoffs."""
        gt = self.results['resource_estimation']['grade_tonnage_curve']
        cutoffs = np.array(gt['cutoff_grades']) * 100
        tonnages = np.array(gt['tonnages'])
        avg_grades = np.array(gt['average_grades']) * 100

        fig, ax1 = plt.subplots(figsize=(7, 5))
        color1, color2 = '#1565C0', '#E65100'

        ax1.plot(cutoffs, tonnages / 1e6, 'o-', color=color1, markersize=4,
                 label='Tonnage')
        ax1.set_xlabel('Cutoff Grade (% U₃O₈)')
        ax1.set_ylabel('Tonnage (Mt)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.fill_between(cutoffs, tonnages / 1e6, alpha=0.1, color=color1)

        ax2 = ax1.twinx()
        ax2.plot(cutoffs, avg_grades, 's--', color=color2, markersize=4,
                 label='Average Grade')
        ax2.set_ylabel('Average Grade (% U₃O₈)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                   framealpha=0.9)
        ax1.set_title('Grade–Tonnage Curve')
        self._save_fig(fig, 'grade_tonnage_curve')

    def _plot_resource_classification(self):
        """Pie chart of resource classification (Measured/Indicated/Inferred)."""
        rc = self.results['resource_estimation']['resource_classification']
        measured = rc.get('measured_tonnage', 0)
        indicated = rc.get('indicated_tonnage', 0)
        inferred = rc.get('inferred_tonnage', 0)

        values = [measured, indicated, inferred]
        labels = ['Measured', 'Indicated', 'Inferred']
        colors = ['#2E7D32', '#F9A825', '#C62828']
        explode = (0.03, 0.03, 0.03)

        # Filter out zero categories
        nonzero = [(v, l, c, e) for v, l, c, e in zip(values, labels, colors, explode) if v > 0]
        if not nonzero:
            return
        values, labels, colors, explode = zip(*nonzero)

        fig, ax = plt.subplots(figsize=(6, 5))
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors, explode=explode,
            autopct='%1.1f%%', startangle=140, pctdistance=0.75,
            textprops={'fontsize': 11})
        for t in autotexts:
            t.set_fontweight('bold')
        ax.set_title('Resource Classification (JORC/NI 43-101)')
        self._save_fig(fig, 'resource_classification')

    # ── Extraction Efficiency ───────────────────────────────────────────

    def _plot_extraction_recovery(self):
        """Uranium recovery curve over time."""
        ee = self.results['extraction_efficiency']['transport_results']
        time_s = np.array(ee['time'])
        recovery = np.array(ee['uranium_recovery'])
        min_len = min(len(time_s), len(recovery))
        time_s = time_s[:min_len]
        recovery = recovery[:min_len]
        time_days = time_s / 86400.0

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(time_days, recovery, color='#1565C0', linewidth=2)
        ax.fill_between(time_days, recovery, alpha=0.1, color='#1565C0')

        # Mark 50% and 90% recovery points
        final_rec = recovery[-1]
        for frac, style, clr in [(0.5, '--', '#FF9800'), (0.9, ':', '#4CAF50')]:
            target = frac * final_rec
            idx = np.searchsorted(recovery, target)
            if 0 < idx < len(time_days):
                ax.axhline(target, color=clr, linestyle=style, alpha=0.6)
                ax.axvline(time_days[idx], color=clr, linestyle=style, alpha=0.6)
                ax.annotate(f'{frac:.0%} recovery\n({time_days[idx]:.0f} d)',
                            xy=(time_days[idx], target),
                            xytext=(time_days[idx] + 20, target * 1.05),
                            fontsize=9, color=clr,
                            arrowprops=dict(arrowstyle='->', color=clr, lw=1))

        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cumulative Uranium Recovery (kg)')
        ax.set_title('In-Situ Leaching: Uranium Recovery Profile')
        self._save_fig(fig, 'extraction_recovery')

    def _plot_concentration_profiles(self):
        """Leachant and uranium concentration spatial profiles."""
        ee = self.results['extraction_efficiency']['transport_results']
        leachant = np.array(ee.get('final_leachant_conc', []))
        uranium = np.array(ee.get('final_uranium_conc', []))

        if leachant.size == 0 or uranium.size == 0:
            return

        # Take a 2D slice if 3D
        if leachant.ndim == 3:
            leachant_slice = leachant[:, :, leachant.shape[2] // 2]
            uranium_slice = uranium[:, :, uranium.shape[2] // 2]
        elif leachant.ndim == 2:
            leachant_slice = leachant
            uranium_slice = uranium
        else:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        im1 = ax1.imshow(leachant_slice.T, origin='lower', cmap='YlOrRd',
                         aspect='auto')
        ax1.set_title('Leachant Concentration')
        ax1.set_xlabel('X (grid cells)')
        ax1.set_ylabel('Y (grid cells)')
        plt.colorbar(im1, ax=ax1, label='Concentration (mol/L)')

        im2 = ax2.imshow(uranium_slice.T, origin='lower', cmap='YlGnBu',
                         aspect='auto')
        ax2.set_title('Dissolved Uranium Concentration')
        ax2.set_xlabel('X (grid cells)')
        ax2.set_ylabel('Y (grid cells)')
        plt.colorbar(im2, ax=ax2, label='Concentration (mol/L)')

        fig.suptitle('Spatial Concentration Profiles (Mid-depth Slice)', y=1.02)
        fig.tight_layout()
        self._save_fig(fig, 'concentration_profiles')

    # ── Environmental Impact ────────────────────────────────────────────

    def _plot_dose_pathways(self):
        """Bar chart of radiation dose by exposure pathway."""
        risk = self.results['environmental_impact']['risk_results']
        pathways = risk.get('pathway_doses', {})

        if not pathways:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(pathways.keys())
        doses = [pathways[n] for n in names]

        # Clean up names for display
        display_names = [n.replace('_', ' ').title() for n in names]
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))

        bars = ax.barh(display_names, doses, color=colors, edgecolor='white',
                       height=0.6)

        # Add ICRP dose limit line
        dose_limit = 1e-3  # 1 mSv/yr for public
        ax.axvline(dose_limit, color='red', linestyle='--', linewidth=1.5,
                   label=f'ICRP Public Limit ({dose_limit:.0e} Sv)')

        ax.set_xlabel('Effective Dose (Sv)')
        ax.set_title('Radiation Dose by Exposure Pathway')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_xscale('log')
        fig.tight_layout()
        self._save_fig(fig, 'dose_pathways')

    def _plot_radionuclide_transport(self):
        """Radionuclide concentration vs distance from source."""
        transport = self.results['environmental_impact']['transport_results']
        distances = np.array(transport.get('distances_m', []))

        if distances.size == 0:
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        nuclide_colors = {'U238': '#1565C0', 'Ra226': '#E65100', 'Rn222': '#2E7D32'}

        for nuclide, conc_data in transport.get('concentrations', {}).items():
            conc = np.array(conc_data)
            if conc.ndim > 1:
                conc = conc[-1, :]  # Final timestep
            if len(conc) != len(distances):
                conc = conc[:len(distances)]
            color = nuclide_colors.get(nuclide, '#666666')
            label = nuclide.replace('238', '-238').replace('226', '-226').replace('222', '-222')
            ax.semilogy(distances, np.maximum(conc, 1e-20), color=color,
                        linewidth=2, label=label)

        ax.set_xlabel('Distance from Source (m)')
        ax.set_ylabel('Concentration (Bq/L)')
        ax.set_title('Radionuclide Transport in Groundwater')
        ax.legend(framealpha=0.9)
        self._save_fig(fig, 'radionuclide_transport')

    # ── Mine Planning / Economics ───────────────────────────────────────

    def _plot_economic_evaluation(self):
        """NPV cash flow and key economic metrics."""
        mp = self.results['mine_planning']
        pit = mp['pit_results']['pit_metrics']
        schedule = mp['schedule_results']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Cash flow chart
        annual = schedule.get('annual_schedule', [])
        if annual:
            years = list(range(1, len(annual) + 1))
            revenues = [a.get('revenue', 0) for a in annual]
            costs = [a.get('total_cost', 0) for a in annual]
            profits = [r - c for r, c in zip(revenues, costs)]

            ax1.bar(years, [r / 1e6 for r in revenues], color='#2E7D32',
                    alpha=0.7, label='Revenue')
            ax1.bar(years, [-c / 1e6 for c in costs], color='#C62828',
                    alpha=0.7, label='Costs')
            ax1.plot(years, [p / 1e6 for p in profits], 'ko-', markersize=4,
                     linewidth=1.5, label='Net Profit')
            ax1.axhline(0, color='black', linewidth=0.8)
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Value (M USD)')
            ax1.set_title('Annual Cash Flow')
            ax1.legend(framealpha=0.9)
        else:
            ax1.text(0.5, 0.5, 'No annual schedule data', ha='center',
                     va='center', transform=ax1.transAxes)

        # Key metrics summary
        metrics = {
            'NPV': f"${pit['npv_usd'] / 1e6:,.1f}M",
            'IRR': f"{pit['irr_percent']:.1f}%",
            'Payback': f"{pit['payback_period_years']:.1f} yr",
            'Avg Grade': f"{pit['average_grade_percent']:.2f}%",
        }
        ax2.axis('off')
        table_data = [[k, v] for k, v in metrics.items()]
        table = ax2.table(cellText=table_data, colLabels=['Metric', 'Value'],
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.2, 2.0)
        # Style header
        for j in range(2):
            table[(0, j)].set_facecolor('#1565C0')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                table[(i, j)].set_facecolor('#E3F2FD' if i % 2 == 0 else 'white')
        ax2.set_title('Key Economic Indicators', pad=20)

        fig.suptitle('Mine Economic Evaluation', fontsize=14, y=1.02)
        fig.tight_layout()
        self._save_fig(fig, 'economic_evaluation')

    def _plot_production_schedule(self):
        """Production schedule: tonnage and grade over mine life."""
        schedule = self.results['mine_planning']['schedule_results']
        annual = schedule.get('annual_schedule', [])
        if not annual:
            return

        years = list(range(1, len(annual) + 1))
        tonnages = [a.get('ore_tonnes', 0) for a in annual]
        grades = [a.get('average_grade', 0) * 100 for a in annual]

        fig, ax1 = plt.subplots(figsize=(7, 5))
        color1, color2 = '#1565C0', '#E65100'

        ax1.bar(years, [t / 1e6 for t in tonnages], color=color1, alpha=0.7,
                label='Ore Tonnage')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Ore Mined (Mt)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.plot(years, grades, 's-', color=color2, markersize=5, linewidth=1.5,
                 label='Average Grade')
        ax2.set_ylabel('Average Grade (% U₃O₈)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                   framealpha=0.9)
        ax1.set_title('Production Schedule')
        self._save_fig(fig, 'production_schedule')

    # ── Uncertainty Quantification ──────────────────────────────────────

    def _plot_monte_carlo_histogram(self):
        """Monte Carlo profit distribution with confidence interval."""
        mc = self.results['uncertainty']['monte_carlo']
        samples = np.array(mc['samples'])
        mean_val = mc['mean']
        ci_low, ci_high = mc['confidence_interval']

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(samples, bins=40, color='#5C6BC0', edgecolor='white',
                alpha=0.8, density=True)

        ax.axvline(mean_val, color='#F44336', linestyle='--', linewidth=2,
                   label=f'Mean: ${mean_val:.2f}/t')
        ax.axvspan(ci_low, ci_high, alpha=0.15, color='#4CAF50',
                   label=f'95% CI: [${ci_low:.2f}, ${ci_high:.2f}]')

        ax.set_xlabel('Profit per Tonne (USD/t)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Monte Carlo Simulation (n={len(samples):,})')
        ax.legend(framealpha=0.9)
        self._save_fig(fig, 'monte_carlo_profit')

    def _plot_sensitivity_tornado(self):
        """Tornado chart showing parameter sensitivity (Sobol indices)."""
        sa = self.results['uncertainty']['sensitivity_analysis']

        if not isinstance(sa, dict) or not sa:
            return

        params = list(sa.keys())
        indices = [sa[p] for p in params]

        # Sort by absolute sensitivity
        sorted_pairs = sorted(zip(params, indices), key=lambda x: abs(x[1]))
        params, indices = zip(*sorted_pairs)

        display_names = [p.replace('_', ' ').title() for p in params]

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ['#C62828' if v > 0.2 else '#F9A825' if v > 0.1 else '#2E7D32'
                  for v in indices]
        bars = ax.barh(display_names, indices, color=colors, edgecolor='white',
                       height=0.6)

        ax.set_xlabel('Sensitivity Index (Sobol)')
        ax.set_title('Parameter Sensitivity Analysis')
        ax.set_xlim(0, max(indices) * 1.15)

        for bar, val in zip(bars, indices):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=10)

        fig.tight_layout()
        self._save_fig(fig, 'sensitivity_tornado')

    # ── Helpers ──────────────────────────────────────────────────────────

    def _save_fig(self, fig, name):
        """Save figure as PNG and store for PDF compilation."""
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        self.figures.append((fig, name))
        print(f"    Saved: {path.name}")

    def _save_combined_pdf(self) -> str:
        """Compile all figures into a single PDF report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = self.output_dir / f"mining_simulation_report_{timestamp}.pdf"

        with PdfPages(str(pdf_path)) as pdf:
            # Title page
            title_fig = plt.figure(figsize=(11, 8.5))
            title_fig.text(0.5, 0.6, 'Uranium Mining Simulation', fontsize=28,
                           ha='center', va='center', fontweight='bold',
                           fontfamily='serif')
            title_fig.text(0.5, 0.5, 'Comprehensive Analysis Report', fontsize=18,
                           ha='center', va='center', fontfamily='serif',
                           color='#555555')
            title_fig.text(0.5, 0.38,
                           datetime.now().strftime('%B %d, %Y'),
                           fontsize=14, ha='center', va='center',
                           fontfamily='serif', color='#777777')
            title_fig.text(0.5, 0.25,
                           'Generated by Nuclear Fuel Cycle Simulation Framework',
                           fontsize=11, ha='center', va='center',
                           fontfamily='serif', color='#999999')
            pdf.savefig(title_fig, facecolor='white')
            plt.close(title_fig)

            # All figures
            for fig, name in self.figures:
                pdf.savefig(fig, facecolor='white', bbox_inches='tight')

        return str(pdf_path)
