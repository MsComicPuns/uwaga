import pandas as pd
import numpy as np
from scipy import stats
import glob
import os

# ─────────────────────────────────────────
#  WCZYTYWANIE DANYCH
# ─────────────────────────────────────────

def load_data(pattern='data/result_*.csv'):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f'Brak plikow pasujacych do: {pattern}')
    df = pd.concat([pd.read_csv(f, sep=';') for f in files], ignore_index=True)
    print(f'Wczytano {len(df)} prob od {df["id_badanego"].nunique()} badanych')
    return df

def preprocess(df):
    # tylko glowne proby (nie trening)
    df = df[df['czy_trening'] == 0].copy()
    # tylko poprawne odpowiedzi do analizy RT
    df_correct = df[df['czy_poprawna'] == 1].copy()
    # usun outliery RT (< 200ms lub > 2000ms)
    df_correct = df_correct[(df_correct['czas_reakcji_ms'] >= 200) &
                            (df_correct['czas_reakcji_ms'] <= 2000)]
    return df, df_correct

# ─────────────────────────────────────────
#  STATYSTYKI OPISOWE
# ─────────────────────────────────────────

def descriptive_stats(df_correct, df_all):
    print('\n' + '='*60)
    print('STATYSTYKI OPISOWE')
    print('='*60)

    # RT per warunek
    rt = df_correct.groupby(['load_condition', 'icon_category'])['czas_reakcji_ms'].agg(
        n='count',
        mean=lambda x: round(x.mean(), 1),
        median=lambda x: round(x.median(), 1),
        sd=lambda x: round(x.std(), 1),
        min='min',
        max='max'
    )
    print('\nCzas reakcji (ms) — poprawne proby:')
    print(rt.to_string())

    # Accuracy per warunek
    acc = df_all.groupby(['load_condition', 'icon_category'])['czy_poprawna'].agg(
        n='count',
        accuracy=lambda x: round(x.mean() * 100, 1)
    )
    print('\nDokladnosc (% poprawnych):')
    print(acc.to_string())

    return rt, acc

# ─────────────────────────────────────────
#  TESTY STATYSTYCZNE
# ─────────────────────────────────────────

def mann_whitney(a, b, label_a, label_b, measure='RT'):
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    med_a, med_b = np.median(a), np.median(b)
    r = u / (len(a) * len(b))  # effect size r
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  {label_a} vs {label_b}: U={u:.0f}, p={p:.4f} {sig} | '
          f'med={med_a:.1f} vs {med_b:.1f} | r={r:.3f}')

def wilcoxon_test(a, b, label_a, label_b):
    # test Wilcoxona dla par (ten sam badany)
    w, p = stats.wilcoxon(a, b)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  {label_a} vs {label_b}: W={w:.0f}, p={p:.4f} {sig}')

def run_tests(df_correct, df_all):
    print('\n' + '='*60)
    print('TESTY STATYSTYCZNE')
    print('='*60)

    rt = df_correct['czas_reakcji_ms']
    load = df_correct['load_condition']
    cat = df_correct['icon_category']

    # 1. Efekt load (LL vs HL)
    print('\n1. Efekt obciazenia poznawczego (LL vs HL) — RT:')
    mann_whitney(
        df_correct[load == 'LL']['czas_reakcji_ms'],
        df_correct[load == 'HL']['czas_reakcji_ms'],
        'LL', 'HL'
    )

    # 2. Efekt kategorii ikony (social vs neutral)
    print('\n2. Efekt kategorii ikony (social vs neutral) — RT:')
    mann_whitney(
        df_correct[cat == 'social']['czas_reakcji_ms'],
        df_correct[cat == 'neutral']['czas_reakcji_ms'],
        'social', 'neutral'
    )

    # 3. Porownanie wszystkich 4 warunkow
    print('\n3. Porownanie warunkow:')
    conds = {
        'LL+neutral': df_correct[(load=='LL') & (cat=='neutral')]['czas_reakcji_ms'],
        'LL+social':  df_correct[(load=='LL') & (cat=='social')]['czas_reakcji_ms'],
        'HL+neutral': df_correct[(load=='HL') & (cat=='neutral')]['czas_reakcji_ms'],
        'HL+social':  df_correct[(load=='HL') & (cat=='social')]['czas_reakcji_ms'],
    }
    pairs = [
        ('LL+neutral', 'HL+neutral'),
        ('LL+social',  'HL+social'),
        ('LL+neutral', 'LL+social'),
        ('HL+neutral', 'HL+social'),
        ('LL+neutral', 'HL+social'),
    ]
    for a, b in pairs:
        mann_whitney(conds[a], conds[b], a, b)

    # 4. Kruskal-Wallis (omnibus)
    print('\n4. Test Kruskal-Wallis (wszystkie 4 warunki):')
    h, p = stats.kruskal(*conds.values())
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  H={h:.3f}, p={p:.4f} {sig}')

    # 5. Accuracy — chi-kwadrat
    print('\n5. Dokladnosc — test chi-kwadrat (LL vs HL):')
    ct = pd.crosstab(df_all['load_condition'], df_all['czy_poprawna'])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  chi2={chi2:.3f}, df={dof}, p={p:.4f} {sig}')

    print('\n5b. Dokladnosc — test chi-kwadrat (social vs neutral):')
    ct2 = pd.crosstab(df_all['icon_category'], df_all['czy_poprawna'])
    chi2, p, dof, _ = stats.chi2_contingency(ct2)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  chi2={chi2:.3f}, df={dof}, p={p:.4f} {sig}')

    # 6. Korelacja RT z numerem proby (efekt praktyki)
    print('\n6. Korelacja RT z numerem proby (efekt praktyki):')
    r, p = stats.spearmanr(df_correct['numer_proby'], df_correct['czas_reakcji_ms'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f'  rho={r:.3f}, p={p:.4f} {sig}')

# ─────────────────────────────────────────
#  LEGENDA
# ─────────────────────────────────────────

def print_legend():
    print('\n' + '='*60)
    print('LEGENDA')
    print('='*60)
    print('  ns  = p >= 0.05  (nieistotne)')
    print('  *   = p <  0.05')
    print('  **  = p <  0.01')
    print('  *** = p <  0.001')
    print('  r   = wielkosc efektu (Mann-Whitney r)')
    print('  rho = korelacja Spearmana')

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

if __name__ == '__main__':
    df = load_data()
    df_all, df_correct = preprocess(df)
    descriptive_stats(df_correct, df_all)
    run_tests(df_correct, df_all)
    print_legend()