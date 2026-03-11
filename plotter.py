#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plotter.py – wizualizacja wyników eksperymentu uwaga.py
========================================================
Użycie:
    python plotter.py <plik.xlsx lub plik.csv> [wyniki.pdf]

Generuje JEDEN plik PDF z dwoma stronami/wykresami:
    Wykres 1: 4 subploty – poprawność (%) dla każdego warunku
    Wykres 2: 4 subploty – boxplot czasu reakcji (ms) dla każdego warunku

Wymagania:
    pip install pandas matplotlib openpyxl
"""

import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ==================== WARUNKI ====================

CONDITIONS = [
    ('LL', 'social',  'LL + Social media',  '#4C72B0'),
    ('LL', 'neutral', 'LL + Neutralna',      '#55A868'),
    ('HL', 'social',  'HL + Social media',  '#DD8452'),
    ('HL', 'neutral', 'HL + Neutralna',      '#C44E52'),
]


# ==================== WCZYTANIE DANYCH ====================

def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    elif ext == '.csv':
        try:
            df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig')
            if df.shape[1] == 1:
                df = pd.read_csv(filepath, sep=',', encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
    else:
        raise ValueError(f"Nieobsługiwany format pliku: {ext}")
    return df


def preprocess(df):
    if 'czy_trening' in df.columns:
        df = df[df['czy_trening'] == 0].copy()

    for col in ['czy_poprawna', 'czas_reakcji_ms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['load_condition', 'icon_category']:
        if col not in df.columns:
            print(f"BLAD: brak kolumny '{col}' – upewnij sie, ze plik pochodzi z nowej wersji uwaga.py")
            sys.exit(1)

    return df


def get_participant_id(df):
    if 'id_badanego' in df.columns and len(df) > 0:
        return str(df['id_badanego'].iloc[0])
    return None


# ==================== WYKRES 1: POPRAWNOSC ====================

def fig_accuracy(df, participant_id=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    title = "Poprawnosc odpowiedzi wg warunkow"
    if participant_id:
        title += f"   |   Badany: {participant_id}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    accs = []
    for idx, (load_cond, icon_cat, label, color) in enumerate(CONDITIONS):
        ax = axes[idx]
        mask = (df['load_condition'] == load_cond) & (df['icon_category'] == icon_cat)
        cond = df[mask]

        if len(cond) > 0 and 'czy_poprawna' in cond.columns:
            acc = cond['czy_poprawna'].mean() * 100
            accs.append(acc)
            ax.bar([''], [acc], color=color, alpha=0.82,
                   edgecolor='black', linewidth=0.9, width=0.55)
            ax.text(0, acc + 1.5, f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=13,
                    fontweight='bold', color=color)
            n_total   = len(cond)
            n_correct = int(cond['czy_poprawna'].sum())
            ax.text(0, -6.5, f'n = {n_correct} / {n_total}',
                    ha='center', va='top', fontsize=9, color='gray')
        else:
            ax.text(0.5, 0.5, 'brak\ndanych', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='gray')

        ax.set_title(label, fontsize=11, fontweight='bold', color=color, pad=8)
        ax.set_ylim(0, 115)
        ax.set_xlim(-0.6, 0.6)
        ax.axhline(y=100, color='lightgray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(y=50,  color='gray',      linestyle=':',  linewidth=0.8, alpha=0.5)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if accs:
        ymin = max(0, min(accs) - 12)
        for ax in axes:
            ax.set_ylim(ymin, 115)

    axes[0].set_ylabel('Poprawnosc (%)', fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


# ==================== WYKRES 2: CZAS REAKCJI ====================

def fig_rt(df, participant_id=None):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    title = "Czas reakcji (ms) wg warunkow  -  poprawne proby"
    if participant_id:
        title += f"   |   Badany: {participant_id}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    all_rt = []
    for idx, (load_cond, icon_cat, label, color) in enumerate(CONDITIONS):
        ax = axes[idx]
        mask = (df['load_condition'] == load_cond) & (df['icon_category'] == icon_cat)
        cond = df[mask]

        if 'czas_reakcji_ms' in cond.columns and 'czy_poprawna' in cond.columns:
            rt_data = cond[cond['czy_poprawna'] == 1]['czas_reakcji_ms'].dropna()
        else:
            rt_data = pd.Series(dtype=float)

        if len(rt_data) >= 3:
            all_rt.append(rt_data)
            ax.boxplot(
                rt_data,
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color='black', linewidth=2.5),
                boxprops=dict(facecolor=color, alpha=0.5, linewidth=1.2),
                whiskerprops=dict(linewidth=1.5, linestyle='--'),
                capprops=dict(linewidth=2),
                flierprops=dict(marker='o', markerfacecolor=color,
                               markeredgecolor='gray', markersize=4, alpha=0.5)
            )
            mean_val = rt_data.mean()
            ax.plot(1, mean_val, 'D', color='white', markeredgecolor='black',
                    markersize=9, zorder=5)
            ax.text(1.32, mean_val, f'M={mean_val:.0f}',
                    va='center', fontsize=9, color='black')
            n   = len(rt_data)
            med = rt_data.median()
            ax.text(0.5, 0.02, f'n={n}  med={med:.0f} ms',
                    ha='center', va='bottom', transform=ax.transAxes,
                    fontsize=8.5, color='gray')
        else:
            ax.text(0.5, 0.5, 'za malo\ndanych', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='gray')

        ax.set_title(label, fontsize=11, fontweight='bold', color=color, pad=8)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if all_rt:
        combined = pd.concat(all_rt)
        ymin = max(0, combined.quantile(0.01) - 50)
        ymax = combined.quantile(0.99) + 120
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    axes[0].set_ylabel('Czas reakcji (ms)', fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


# ==================== ZAPIS DO PDF ====================

def save_to_pdf(fig1, fig2, output_path):
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig1, bbox_inches='tight')
        pdf.savefig(fig2, bbox_inches='tight')
        d = pdf.infodict()
        d['Title'] = 'Wyniki eksperymentu - Uwaga wzrokowa'
        d['Author'] = 'plotter.py'
    print(f"PDF zapisany: {output_path}  (2 strony)")


# ==================== STATYSTYKI W KONSOLI ====================

def print_summary(df):
    print("\n" + "=" * 68)
    print("PODSUMOWANIE WYNIKOW")
    print("=" * 68)
    print(f"{'Warunek':<22} {'n':>5} {'Poprawnosc':>12} {'Sr. RT (ms)':>13} {'Med. RT':>9}")
    print("-" * 68)
    for load_cond, icon_cat, label, _ in CONDITIONS:
        mask = (df['load_condition'] == load_cond) & (df['icon_category'] == icon_cat)
        cond = df[mask]
        if len(cond) == 0:
            print(f"{label:<22} {'brak':>5}")
            continue
        n       = len(cond)
        acc     = cond['czy_poprawna'].mean() * 100 if 'czy_poprawna' in cond else float('nan')
        rt_ok   = cond[cond['czy_poprawna'] == 1]['czas_reakcji_ms'].dropna()
        mean_rt = rt_ok.mean()   if len(rt_ok) else float('nan')
        med_rt  = rt_ok.median() if len(rt_ok) else float('nan')
        print(f"{label:<22} {n:>5} {acc:>11.1f}% {mean_rt:>12.1f} {med_rt:>9.1f}")
    print("=" * 68 + "\n")


# ==================== MAIN ====================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'wykres_wynikow.pdf'

    if not output_file.lower().endswith('.pdf'):
        output_file += '.pdf'

    if not os.path.exists(input_file):
        print(f"Blad: plik '{input_file}' nie istnieje.")
        sys.exit(1)

    print(f"Wczytywanie: {input_file}")
    df = load_data(input_file)
    print(f"Wczytano {len(df)} wierszy, {df.shape[1]} kolumn.")

    df  = preprocess(df)
    pid = get_participant_id(df)

    print_summary(df)

    fig1 = fig_accuracy(df, pid)
    fig2 = fig_rt(df, pid)

    save_to_pdf(fig1, fig2, output_file)
    plt.close('all')


if __name__ == '__main__':
    main()
