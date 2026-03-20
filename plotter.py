import sys, os, glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.backends.backend_pdf import PdfPages

CONDITIONS = [
    ('LL', 'social',  'LL + Social media', '#4C72B0'),
    ('LL', 'neutral', 'LL + Neutralna',    '#55A868'),
    ('HL', 'social',  'HL + Social media', '#DD8452'),
    ('HL', 'neutral', 'HL + Neutralna',    '#C44E52'),
]

def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(filepath)
    try:
        df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig')
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, sep=',', encoding='utf-8-sig')
        return df
    except Exception:
        return pd.read_csv(filepath, encoding='utf-8-sig')

def preprocess(df):
    if 'czy_trening' in df.columns:
        df = df[df['czy_trening'] == 0].copy()
    for col in ['czy_poprawna', 'czas_reakcji_ms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['load_condition', 'icon_category']:
        if col not in df.columns:
            raise ValueError(f"Brak kolumny '{col}' – uzyj pliku z nowej wersji uwaga.py")
    return df

def make_fig_accuracy(df, pid=None):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Poprawnosc odpowiedzi" + (f"  |  Badany: {pid}" if pid else ""),
                 fontsize=13, fontweight='bold')
    accs = []
    for idx, (lc, ic, label, color) in enumerate(CONDITIONS):
        ax = axes[idx]
        cond = df[(df['load_condition'] == lc) & (df['icon_category'] == ic)]
        if len(cond) > 0 and 'czy_poprawna' in cond.columns:
            acc = cond['czy_poprawna'].mean() * 100
            accs.append(acc)
            ax.bar([''], [acc], color=color, alpha=0.82, edgecolor='black', linewidth=0.9, width=0.55)
            ax.text(0, acc + 1.5, f'{acc:.1f}%', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color=color)
            ax.text(0, -6, f'n={int(cond["czy_poprawna"].sum())}/{len(cond)}',
                    ha='center', va='top', fontsize=9, color='gray')
        else:
            ax.text(0.5, 0.5, 'brak\ndanych', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
        ax.set_title(label, fontsize=10, fontweight='bold', color=color, pad=7)
        ax.set_ylim(0, 115)
        ax.set_xlim(-0.6, 0.6)
        ax.axhline(100, color='lightgray', ls='--', lw=0.8)
        ax.axhline(50,  color='gray',      ls=':',  lw=0.8)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if accs:
        for ax in axes:
            ax.set_ylim(max(0, min(accs) - 12), 115)
    axes[0].set_ylabel('Poprawnosc (%)', fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig

def make_fig_rt(df, pid=None):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Czas reakcji (ms) – poprawne proby" + (f"  |  Badany: {pid}" if pid else ""),
                 fontsize=13, fontweight='bold')
    all_rt = []
    for idx, (lc, ic, label, color) in enumerate(CONDITIONS):
        ax = axes[idx]
        cond = df[(df['load_condition'] == lc) & (df['icon_category'] == ic)]
        rt_data = pd.Series(dtype=float)
        if 'czas_reakcji_ms' in cond.columns and 'czy_poprawna' in cond.columns:
            rt_data = cond[cond['czy_poprawna'] == 1]['czas_reakcji_ms'].dropna()
        if len(rt_data) >= 3:
            all_rt.append(rt_data)
            ax.boxplot(rt_data, patch_artist=True, widths=0.5,
                       medianprops=dict(color='black', linewidth=2.5),
                       boxprops=dict(facecolor=color, alpha=0.5, linewidth=1.2),
                       whiskerprops=dict(linewidth=1.5, linestyle='--'),
                       capprops=dict(linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor=color,
                                       markeredgecolor='gray', markersize=4, alpha=0.5))
            mv = rt_data.mean()
            ax.plot(1, mv, 'D', color='white', markeredgecolor='black', markersize=8, zorder=5)
            ax.text(1.32, mv, f'M={mv:.0f}', va='center', fontsize=9)
            ax.text(0.5, 0.02, f'n={len(rt_data)}  med={rt_data.median():.0f} ms',
                    ha='center', va='bottom', transform=ax.transAxes, fontsize=8.5, color='gray')
        else:
            ax.text(0.5, 0.5, 'za malo\ndanych', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
        ax.set_title(label, fontsize=10, fontweight='bold', color=color, pad=7)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if all_rt:
        combined = pd.concat(all_rt)
        for ax in axes:
            ax.set_ylim(max(0, combined.quantile(0.01) - 50), combined.quantile(0.99) + 120)
    axes[0].set_ylabel('Czas reakcji (ms)', fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig

def build_accuracy_table(datasets):
    rows = []
    for pid, df in datasets:
        row = {'participant': pid}
        for lc, ic, label, _ in CONDITIONS:
            cond = df[(df['load_condition'] == lc) & (df['icon_category'] == ic)]
            row[label] = cond['czy_poprawna'].mean() * 100 if len(cond) else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)

def make_participant_colors(n):
    if n <= 0:
        return []
    return [hsv_to_rgb((idx / n, 0.7, 0.85)) for idx in range(n)]

def make_fig_accuracy_spaghetti(acc_table):
    labels = [label for _, _, label, _ in CONDITIONS]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("Poprawnosc odpowiedzi - spaghetti plot", fontsize=13, fontweight='bold')

    colors = make_participant_colors(len(acc_table))
    for idx, (_, row) in enumerate(acc_table.iterrows()):
        ax.plot(
            x,
            row[labels].tolist(),
            color=colors[idx],
            linewidth=1.6,
            marker='o',
            markersize=5,
            alpha=0.9
        )

    mean_values = acc_table[labels].mean(axis=0, skipna=True)
    ax.plot(
        x,
        mean_values,
        color='black',
        linewidth=3.2,
        linestyle='dashdot',
        marker='o',
        markersize=7,
        label='Srednia grupy',
        zorder=10
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha='right')
    ax.set_ylabel('Poprawnosc (%)', fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(100, color='lightgray', ls='--', lw=0.8)
    ax.axhline(50, color='gray', ls=':', lw=0.8)
    ax.grid(axis='y', color='0.88', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig

def save_pdf(*figs, output_path):
    with PdfPages(output_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
    print(f"PDF zapisany: {output_path}")

def find_data_files():
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if not os.path.exists(data_dir):
        print(f"Blad: brak folderu data/ w {data_dir}")
        sys.exit(1)
    files = sorted(glob.glob(os.path.join(data_dir, 'result_*.csv')))
    if not files:
        print(f"Blad: brak plikow result_*.csv w {data_dir}")
        sys.exit(1)
    return files

def print_summary(df):
    print("\n" + "=" * 66)
    print("PODSUMOWANIE")
    print("=" * 66)
    print(f"{'Warunek':<22} {'n':>5} {'Poprawnosc':>12} {'Sr.RT':>10} {'Med.RT':>9}")
    print("-" * 66)
    for lc, ic, label, _ in CONDITIONS:
        cond = df[(df['load_condition'] == lc) & (df['icon_category'] == ic)]
        if not len(cond):
            print(f"{label:<22}  brak danych")
            continue
        acc  = cond['czy_poprawna'].mean() * 100
        rt   = cond[cond['czy_poprawna'] == 1]['czas_reakcji_ms'].dropna()
        print(f"{label:<22} {len(cond):>5} {acc:>11.1f}% "
              f"{rt.mean():>9.1f} {rt.median():>9.1f}")
    print("=" * 66 + "\n")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
PLOTS_DIR  = os.path.join(SCRIPT_DIR, 'plots')

def get_data_files():
    if not os.path.exists(DATA_DIR):
        print(f"Blad: brak folderu data/ ({DATA_DIR})")
        sys.exit(1)
    files = sorted([f for f in os.listdir(DATA_DIR)
                    if f.startswith('result_') and f.endswith('.csv')])
    if not files:
        print(f"Brak plikow result_*.csv w {DATA_DIR}")
        sys.exit(1)
    return files

def pick_file():
    files = get_data_files()
    print("\nDostepne pliki wynikow:")
    print("  [0] Wszystkie pliki -> zapisz do plots/")
    for i, fname in enumerate(files):
        print(f"  [{i+1}] {fname}")
    choice = input("\nWybierz numer: ").strip()
    if choice == '0':
        return None
    if not choice.isdigit():
        print("Nieprawidlowy wybor.")
        sys.exit(1)

    idx = int(choice) - 1
    if idx < 0 or idx >= len(files):
        print("Nieprawidlowy wybor.")
        sys.exit(1)

    return os.path.join(DATA_DIR, files[idx])

def process_file(input_file, output_file):
    print(f"\nWczytuję: {os.path.basename(input_file)}")
    df  = preprocess(load_data(input_file))
    pid = str(df['id_badanego'].iloc[0]) if 'id_badanego' in df.columns and len(df) else None
    print(f"Prob: {len(df)}  |  badany: {pid or '?'}")
    print_summary(df)
    fig1 = make_fig_accuracy(df, pid)
    fig2 = make_fig_rt(df, pid)
    save_pdf(fig1, fig2, output_path=output_file)
    plt.close('all')
    return pid, df

def main():
    if len(sys.argv) >= 2:
        process_file(sys.argv[1],
                     sys.argv[2] if len(sys.argv) >= 3
                     else os.path.splitext(sys.argv[1])[0] + '_wykres.pdf')
        return

    choice = pick_file()

    if choice is None:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        files = get_data_files()
        print(f"\nPrzetwarzam {len(files)} plikow -> {PLOTS_DIR}")
        datasets = []
        for fname in files:
            input_file  = os.path.join(DATA_DIR, fname)
            output_file = os.path.join(PLOTS_DIR, os.path.splitext(fname)[0] + '_wykres.pdf')
            try:
                datasets.append(process_file(input_file, output_file))
            except Exception as e:
                print(f"  BLAD ({fname}): {e}")
        if datasets:
            acc_table = build_accuracy_table(datasets)
            fig = make_fig_accuracy_spaghetti(acc_table)
            group_output = os.path.join(PLOTS_DIR, 'group_accuracy_spaghetti.pdf')
            save_pdf(fig, output_path=group_output)
            plt.close(fig)
        print(f"\nGotowe. Wykresy zapisane w: {PLOTS_DIR}")
    else:
        output_file = os.path.splitext(choice)[0] + '_wykres.pdf'
        process_file(choice, output_file)

if __name__ == '__main__':
    main()
