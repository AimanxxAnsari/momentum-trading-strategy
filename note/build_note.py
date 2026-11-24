"""Build the research note (note/research-note.pdf) from the tables in results/.

    python note/build_note.py

Every number in the note is read from results/*.csv (or, for the T-bill average, from the factor
data), so re-running the notebooks and then this script keeps the note in step with the analysis.
Needs pdflatex with booktabs, graphicx, amsmath, hyperref, lmodern, xcolor, tabularx and float.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
OUT = ROOT / "note"

AUTHOR = "AimanxxAnsari"          # replace with your name
REPO = "https://github.com/AimanxxAnsari/momentum-trading-strategy"


def csv(name, **kw):
    return pd.read_csv(RES / f"{name}.csv", **kw)


def pct(x, nd=1, sign=False):
    s = f"{100 * x:+.{nd}f}" if sign else f"{100 * x:.{nd}f}"
    return s.replace("-", "$-$") + r"\%"


def num(x, nd=2, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


# -- numbers ------------------------------------------------------------------------------------
V = {}
b5 = csv("phase5_backtest_2010_2023", index_col=0)
base, sp, top = b5.loc["Baseline (momentum + ARIMA views)"], b5.loc["Cap-weight S&P 500"], b5.loc["Cap-weight top 50"]
ms_hist = b5.loc["Grinold alpha: historical mean"]
ms_mom = b5.loc["Grinold alpha: momentum 12-1"]

ic = csv("phase5_ic").set_index(["universe", "signal", "horizon"])
sub = csv("phase5_ic_subperiods", index_col=[0, 1])
quint = csv("phase5_quintiles", index_col=[0, 1])
fm = csv("phase5_fama_macbeth", index_col=[0, 1])
diag5 = csv("phase5_alpha_diagnostics_2010_2023", index_col=0)

p6 = csv("phase6_main_2010_2023", index_col=0)
act = p6.loc["Active: historical mean"]
p7 = csv("phase7_portfolios_2010_2023", index_col=0)
fac = csv("phase7_momentum_factor", index_col=[0, 1])
chk = csv("phase7_risk_model_checks", index_col=0)

sh = csv("phase8_sharpe_tbill", index_col=0)
att = csv("phase8_attribution_vs_sp500", index_col=0)
att_all = csv("phase8_attribution_all_trials", index_col=0)
boot = csv("phase8_sharpe_difference_bootstrap", index_col=0)
dsr = csv("phase8_deflated_sharpe", index_col=[0, 1])
grp = csv("phase8_overfitting_by_group", index_col=0)

from mts import factors  # noqa: E402
rf_avg = factors.load().loc["2010-01-04":"2023-12-29", "RF"].mean() * 252

best_ir = dsr.loc[[i for i in dsr.index if i[0].startswith("best trial")]]
dsr_vals = best_ir["deflated Sharpe"]
hm_sub = sub.loc[("Top 50", "hist_mean")]
first = lambda s: s.split(" ")[0]
bp = boot["p"].astype(float)


def boot_row(name):
    return boot.loc[[i for i in boot.index if i.startswith(f"{name}  vs  Cap-weight S&P 500")][0]]


# -- tables ---------------------------------------------------------------------------------------
def table_baseline():
    rows = [("Strategy (baseline)", base, sh.loc["Baseline", "Sharpe, T-bill"]),
            ("Cap-weighted top 50", top, sh.loc["Cap-weight top 50", "Sharpe, T-bill"]),
            ("Cap-weighted S\\&P 500 proxy", sp, sh.loc["Cap-weight S&P 500", "Sharpe, T-bill"])]
    body = "\n".join(
        f"{n} & {pct(r['CAGR'])} & {pct(r['Volatility'])} & {num(r['Sharpe'])} & {num(t)} & {pct(r['Max drawdown'])} & {pct(r['Turnover / yr'], 0)} \\\\"
        for n, r, t in rows)
    return body


def table_construction():
    rows = [("Baseline (momentum, ARIMA views, max-Sharpe)", base["Sharpe"], base["Turnover / yr"], base["Max drawdown"]),
            ("Max-Sharpe with Grinold alpha, historical mean", ms_hist["Sharpe"], ms_hist["Turnover / yr"], ms_hist["Max drawdown"]),
            ("Max-Sharpe with Grinold alpha, momentum", ms_mom["Sharpe"], ms_mom["Turnover / yr"], ms_mom["Max drawdown"]),
            ("Benchmark-relative, historical mean", act["Sharpe"], act["Turnover / yr"], act["Max drawdown"]),
            ("Benchmark-relative, momentum", p6.loc["Active: momentum 12-1", "Sharpe"], p6.loc["Active: momentum 12-1", "Turnover / yr"],
             p6.loc["Active: momentum 12-1", "Max drawdown"]),
            ("Cap-weighted top 50", top["Sharpe"], top["Turnover / yr"], top["Max drawdown"])]
    return "\n".join(f"{n} & {num(s)} & {pct(t, 0)} & {pct(d)} \\\\" for n, s, t, d in rows)


def table_inference():
    rows = [("Baseline", "Baseline"), ("Best by Sharpe: max weight 10\\%", "Baseline, max weight 10%"),
            ("Best by IR: ARIMA views corrected", "Baseline, ARIMA views corrected to mean"),
            ("Benchmark-relative, historical mean", "Active: historical mean")]
    out = []
    for label, k in rows:
        b = boot_row(k)
        a = att.loc[k]
        out.append(f"{label} & {num(sh.loc[k, 'Sharpe, T-bill'], 3)} & {num(b['difference'], 3, True)} & {num(b['p'])} & "
                   f"{pct(a['alpha (annual)'], 2, True)} ({num(a['alpha t'], 2, True)}) & {num(a['Mom'], 2, True)} & {num(a['CMA'], 2, True)} \\\\")
    return "\n".join(out)


# -- document --------------------------------------------------------------------------------------
TEX = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,graphicx,amsmath,xcolor,tabularx,float}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue!60!black,citecolor=black]{hyperref}
\graphicspath{{../images/}}
\setlength{\parskip}{3pt}
\renewcommand{\arraystretch}{1.12}
\newcommand{\notecaption}[1]{\par\smallskip{\small #1}\par}
\title{\vspace{-1.2cm}\textbf{Does a forecast-driven Black--Litterman overlay add value to large-cap momentum?}\\[2pt]
\large Evidence from the point-in-time S\&P 500, 2003--2023}
\author{<<AUTHOR>>\\\small\url{<<REPO>>}}
\date{Research note}
\begin{document}
\maketitle
\vspace{-0.6cm}
\begin{abstract}
\noindent A long-only strategy selects the 25 strongest 12--1 month momentum stocks among the 50 largest S\&P 500
members, turns ARIMA return forecasts into Black--Litterman views and holds a max-Sharpe portfolio, rebalanced
quarterly with transaction costs. Over 2010--2023 it returned <<b_cagr>> a year against <<sp_cagr>> for an
S\&P 500 proxy, but the gap is risk and factor exposure, not skill. Signal research finds no momentum premium in the
S\&P 500 over 2003--2023; the one signal with predictive power in the top 50 (each stock's historical mean return)
comes from the 2017--2023 growth rally. Fixing the portfolio construction with Grinold alphas, a factor risk model
and a benchmark-relative optimiser cuts turnover from <<b_to>> to <<a_to>> a year and leaves the strategy earning
about its benchmark. With the actual T-bill rate the strategy's Sharpe ratio is <<b_sh_tb>> against <<sp_sh_tb>>.
Relative to the index it is a momentum-and-growth tilt; across all 47 variants tested, factor alpha t-statistics
range from <<alpha_t_min>> to <<alpha_t_max>>, and the best variant's deflated Sharpe ratio is <<dsr_lo>>--<<dsr_hi>>,
short of the usual 0.95.
\end{abstract}

\section{Data and design}
\textbf{Data.} Daily Bloomberg prices, market capitalisations and book values for every company that was in the
S\&P 500 at any point in 2000--2023, and 168 quarterly constituent snapshots. The universe on each date is the 50
largest members of the most recent snapshot, so there is no survivorship bias. Three properties of the data shape
the analysis: prices are split-adjusted but there are no dividends (all returns are price returns); the earnings
field is empty, so value is measured by book-to-price only; and book value is stamped at the fiscal period end,
before publication, so it is lagged three months. Fama--French factors and the 1-month T-bill rate come from Kenneth
French's data library. The T-bill averaged <<rf>> a year over 2010--2023, against the constant 3\% assumed in
Phases~1--7.

\textbf{Design.} Every decision uses only data available on the rebalance date (the last trading day of each
quarter); holdings drift between rebalances; every portfolio, benchmarks included, pays 10\,bps per unit traded.
The analysis was built up in eight phases, each reported in its own notebook; the code is a tested Python package
(37 unit tests on synthetic data). All 47 variants of the strategy that were run are kept and corrected for in
Section~\ref{sec:inference}.

\section{The baseline strategy}
Black--Litterman combines a market-capitalisation prior $\pi = \delta\Sigma w_{\text{mkt}}$ with one absolute view per
stock (the ARIMA forecast minus the risk-free rate) and He--Litterman uncertainty $\Omega = \operatorname{diag}(\tau\Sigma)$,
with a Ledoit--Wolf covariance. The optimiser maximises the posterior Sharpe ratio, long-only, at most 20\% per stock.

\begin{table}[H]\centering\small
\notecaption{\textbf{Table 1.} Baseline, 2010--2023, walk-forward, after costs, price returns.}
\begin{tabular}{lrrrrrr}\toprule
& CAGR & Volatility & Sharpe (3\%) & Sharpe (T-bill) & Max drawdown & Turnover / yr\\\midrule
<<table_baseline>>
\bottomrule\end{tabular}
\end{table}
\vspace{-0.3cm}
The strategy's information ratio against the S\&P 500 proxy is <<b_ir>> (t = <<b_t>>). Adding the layers one at a time
shows that momentum selection, the optimiser and the ARIMA views each move the Sharpe ratio by 0.03 or less, while
turnover rises about sevenfold. Two diagnostics explain why: 63\% of the fitted ARIMA models are (0,0,0), so the
``forecasts'' are mostly each stock's historical mean return; and the optimiser concentrates the portfolio into a
median of 7 names at the 20\% cap.

\begin{figure}[H]\centering\includegraphics[width=0.66\textwidth]{equity_curve.png}
\notecaption{\textbf{Figure 1.} Growth of \$1, 2010--2023, after costs (from \texttt{main.ipynb}).}\end{figure}

\section{Is there a premium to capture?}
Eight signals were tested on their own, monthly over 2003--2023, for the whole point-in-time S\&P 500 and its 50
largest members: rank information coefficients (IC) at 1--12 month horizons with Newey--West t-statistics, three
sub-periods, quintile portfolios and Fama--MacBeth regressions.
\begin{itemize}\setlength\itemsep{1pt}
\item \textbf{No momentum premium.} For 12--1 momentum in the S\&P 500 the 3-month IC is <<ic_mom>> (t = <<ic_mom_t>>) and
the top-minus-bottom quintile earned <<q_mom>> a year (t = <<q_mom_t>>). No signal is significant in the full index.
\item \textbf{The historical-mean signal is the growth rally.} It is the only significant signal in the top 50 (3-month
IC <<ic_hm>>, t = <<ic_hm_t>>), but its IC is <<hm_1>>, <<hm_2>> and <<hm_3>> in 2003--09, 2010--16 and 2017--23; with the other
signals as controls its Fama--MacBeth t-statistic is <<fm_hm_t>>. It is correlated $-0.35$ with book-to-price.
\end{itemize}

\begin{figure}[H]\centering\includegraphics[width=0.84\textwidth]{phase5_ic.png}
\notecaption{\textbf{Figure 2.} Mean 3-month rank IC by signal, 2003--2023, with 95\% intervals.}\end{figure}

\section{Fixing the portfolio construction}
Signals were turned into expected returns with Grinold's rule, $\alpha_i = \text{IC}\times\omega_i\times z_i$, the IC
estimated only from past data and $\omega_i$ each stock's specific volatility from a statistical factor risk model
(five principal components). A benchmark-relative optimiser then maximises
$h\,\alpha^\top(w-b) - h\,\lambda\,(w-b)^\top\Sigma(w-b) - c\,\lVert w-w_{\text{current}}\rVert_1$ against the
cap-weighted top 50, with $\lambda = 10$ set from Grinold's rule for the optimal tracking error and trades charged
at their real cost.

\begin{table}[H]\centering\small
\notecaption{\textbf{Table 2.} Portfolio construction, 2010--2023, after costs (Sharpe ratios with the constant 3\%).}
\begin{tabular}{lrrr}\toprule
& Sharpe & Turnover / yr & Max drawdown\\\midrule
<<table_construction>>
\bottomrule\end{tabular}
\end{table}
\vspace{-0.3cm}
With the same alpha, max-Sharpe turns a historical-mean signal into <<ms_to>> turnover and a <<ms_dd>> drawdown, and a
momentum alpha whose cross-sectional standard deviation is under 1\% a year into <<msm_to>> turnover. The
benchmark-relative optimiser holds 44--48 names, transfers the signal to the weights (transfer coefficient <<tc_hm>>
for the historical mean), and earns <<a_act>> a year over its benchmark at <<a_te>> tracking error (IR <<a_ir>>, t = <<a_t>>). The construction
problem is solved; what it reveals is that the alphas are worth very little.

Two standard momentum refinements were also tested. Volatility management (Barroso and Santa-Clara, 2015) transforms
the long-short momentum factor: maximum drawdown <<vm_dd0>> $\rightarrow$ <<vm_dd1>>, worst month <<vm_w0>> $\rightarrow$
<<vm_w1>>, Sharpe ratio <<vm_s0>> $\rightarrow$ <<vm_s1>> (t = <<vm_t>>). Residual momentum (Blitz et al., 2011) is steadier
than 12--1 momentum in the top 50 but has no premium in the S\&P 500. Neither helps the long-only top-50 portfolio.
The risk model's tracking-error forecasts were accurate for random portfolios (bias statistic <<chk_rand>>) and specific
risk (<<chk_spec>>) but too low for optimised ones; adding the alpha as a risk factor (Lee and Stefek, 2008) brings the
historical-mean portfolio from <<bias0>> to <<bias1>>.

\section{Attribution and multiple testing}\label{sec:inference}
All 47 variants were rebuilt over 2010--2023 and tested against the S\&P 500 proxy with the actual T-bill rate.
Attribution is relative to the proxy, because the price data lacks dividends while the Fama--French factors include
them. Sharpe-ratio differences are tested with a circular block bootstrap (21-day blocks, 2,000 resamples).

\begin{table}[H]\centering\small
\notecaption{\textbf{Table 3.} Inference, 2010--2023. Factor alpha and loadings: regression of returns relative to the
S\&P 500 proxy on the Fama--French five factors plus momentum, Newey--West t-statistics.}
\begin{tabular}{lrrrrrr}\toprule
& Sharpe & Difference & Bootstrap & Factor alpha & Momentum & Investment\\
& (T-bill) & vs S\&P 500 & p & a year (t) & loading & (CMA) loading\\\midrule
<<table_inference>>
\bottomrule\end{tabular}
\end{table}
\vspace{-0.3cm}
\begin{itemize}\setlength\itemsep{1pt}
\item \textbf{Factor tilts, not alpha.} The baseline is long momentum (t = <<mom_t>>) and tilted against conservative-investment
and value stocks; across all 47 variants factor alpha t-statistics range from <<alpha_t_min>> to <<alpha_t_max>>. No
Sharpe-ratio difference is significant (smallest p-value <<p_min>>).
\item \textbf{Deflated Sharpe ratio} (Bailey and L\'opez de Prado, 2014). The best information ratio, <<best_ir>>, exceeds the
<<luck47>> that luck alone would give from 47 independent tries; the deflated Sharpe ratio is <<dsr47>>, or <<dsr_mid>>--<<dsr_hi>>
allowing for how correlated the variants are (2--4 effective trials).
\item \textbf{Overfitting} (Bailey et al., 2017). Within each group of variants against its own benchmark, the probability of
backtest overfitting is <<pbo1>> and <<pbo2>>, and the in-sample winner's IR falls from about 0.5 to <<oos1>> and <<oos2>> out of sample.
\end{itemize}

\begin{figure}[H]\centering\includegraphics[width=0.58\textwidth]{phase8_factor_tilts.png}
\notecaption{\textbf{Figure 3.} Factor loadings relative to the S\&P 500 proxy, 2010--2023, with 95\% intervals.}\end{figure}

\begin{figure}[H]\centering\includegraphics[width=0.84\textwidth]{phase8_overfitting.png}
\notecaption{\textbf{Figure 4.} Information ratios of each variant in 2010--2016 and 2017--2023, by group.}\end{figure}

\section{Conclusions}
Large-cap momentum offered no premium in 2003--2023, and the strategy's apparent edge over the S\&P 500 is a
momentum-and-growth tilt that disappears under factor attribution, the actual risk-free rate and corrections for
the number of variants tried. The more durable lessons are methodological: a max-Sharpe optimiser converts weak
forecasts into concentrated, high-turnover bets; a benchmark-relative optimiser with risk and cost penalties
expresses the same forecasts as small, cheap tilts; risk forecasts for optimised portfolios fail when the risk model
does not contain the alpha; and the best of many backtests is weak evidence until deflated.

\textbf{Limitations.} Price returns only (no dividends); no sector data; linear transaction costs; one market and
period (2003--2023); the 47 variants are a lower bound on what was tried. The natural next test is out-of-sample data
from 2024 onwards, with the settings fixed now.

\section*{References}
\footnotesize
\begin{description}\setlength\itemsep{0pt}
\item Bailey, D. H. and L\'opez de Prado, M. (2014). The deflated Sharpe ratio. \emph{Journal of Portfolio Management} 40(5).
\item Bailey, D. H., Borwein, J., L\'opez de Prado, M. and Zhu, Q. J. (2017). The probability of backtest overfitting. \emph{Journal of Computational Finance} 20(4).
\item Barroso, P. and Santa-Clara, P. (2015). Momentum has its moments. \emph{Journal of Financial Economics} 116(1).
\item Black, F. and Litterman, R. (1992). Global portfolio optimization. \emph{Financial Analysts Journal} 48(5).
\item Blitz, D., Huij, J. and Martens, M. (2011). Residual momentum. \emph{Journal of Empirical Finance} 18(3).
\item Clarke, R., de Silva, H. and Thorley, S. (2002). Portfolio constraints and the fundamental law of active management. \emph{Financial Analysts Journal} 58(5).
\item Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. \emph{Journal of Financial Economics} 116(1).
\item Fama, E. F. and MacBeth, J. D. (1973). Risk, return, and equilibrium. \emph{Journal of Political Economy} 81(3).
\item Grinold, R. C. and Kahn, R. N. (2000). \emph{Active Portfolio Management}, 2nd ed. McGraw-Hill.
\item He, G. and Litterman, R. (1999). The intuition behind Black--Litterman model portfolios. Goldman Sachs.
\item Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers. \emph{Journal of Finance} 48(1).
\item Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. \emph{Journal of Multivariate Analysis} 88(2).
\item Lee, J.-H. and Stefek, D. (2008). Do risk factors eat alphas? \emph{Journal of Portfolio Management} 34(4).
\item Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. \emph{Econometrica} 55(3).
\end{description}
\end{document}
"""


def build():
    a_t = att_all["alpha t"].astype(float)
    vm0, vm1 = fac.loc[("12-1 momentum", "raw")], fac.loc[("12-1 momentum", "volatility-managed")]
    fill = {
        "AUTHOR": AUTHOR, "REPO": REPO,
        "b_cagr": pct(base["CAGR"]), "sp_cagr": pct(sp["CAGR"]), "b_to": pct(base["Turnover / yr"], 0),
        "a_to": pct(act["Turnover / yr"], 0), "b_sh_tb": num(sh.loc["Baseline", "Sharpe, T-bill"], 3),
        "sp_sh_tb": num(sh.loc["Cap-weight S&P 500", "Sharpe, T-bill"], 3),
        "alpha_t_min": num(a_t.min(), 2, True), "alpha_t_max": num(a_t.max(), 2, True),
        "dsr_lo": num(dsr_vals.min()), "dsr_hi": num(dsr_vals.max()), "dsr47": num(dsr_vals.iloc[0]),
        "dsr_mid": num(sorted(dsr_vals)[1]), "rf": pct(rf_avg, 2),
        "b_ir": num(base["IR"]), "b_t": num(base["t-stat"]),
        "ic_mom": num(ic.loc[("S&P 500", "mom_12_1", "fwd_3m"), "IC"], 3, True), "ic_mom_t": num(ic.loc[("S&P 500", "mom_12_1", "fwd_3m"), "t"], 2, True),
        "q_mom": pct(quint.loc[("S&P 500", "mom_12_1"), "Q5 − Q1"], 1, True), "q_mom_t": num(quint.loc[("S&P 500", "mom_12_1"), "t"], 2, True),
        "ic_hm": num(ic.loc[("Top 50", "hist_mean", "fwd_3m"), "IC"], 3), "ic_hm_t": num(ic.loc[("Top 50", "hist_mean", "fwd_3m"), "t"], 1),
        "hm_1": first(hm_sub["2003–2009"]), "hm_2": first(hm_sub["2010–2016"]), "hm_3": first(hm_sub["2017–2023"]),
        "fm_hm_t": num(fm.loc[("Top 50", "hist_mean"), "t"], 2),
        "ms_to": pct(ms_hist["Turnover / yr"], 0), "ms_dd": pct(ms_hist["Max drawdown"]), "msm_to": pct(ms_mom["Turnover / yr"], 0),
        "tc_hm": num(csv("phase6_diagnostics_2010_2023", index_col=0).loc["Active: historical mean", "transfer coefficient"]),
        "a_act": pct(act["Active return vs top 50"], 1, True), "a_te": pct(act["Tracking error vs top 50"]),
        "a_ir": num(act["IR vs top 50"]), "a_t": num(act["t-stat vs top 50"]),
        "vm_dd0": pct(vm0["max drawdown"]), "vm_dd1": pct(vm1["max drawdown"]), "vm_w0": pct(vm0["worst month"]), "vm_w1": pct(vm1["worst month"]),
        "vm_s0": num(vm0["Sharpe"]), "vm_s1": num(vm1["Sharpe"]), "vm_t": num(vm1["t (mean)"]),
        "chk_rand": num(chk.loc["Statistical, equal weights", "random active portfolios"]),
        "chk_spec": num(chk.loc["Statistical, equal weights", "specific risk"]),
        "bias0": num(p7.loc["Phase 6: historical mean", "Bias statistic"]), "bias1": num(p7.loc["Historical mean, alpha-aligned risk model", "Bias statistic"]),
        "mom_t": num(att.loc["Baseline", "Mom t"], 1), "p_min": num(bp.min()),
        "best_ir": num(best_ir["IR (annual)"].iloc[0], 3), "luck47": num(best_ir["expected best IR by luck (annual)"].iloc[0], 3),
        "pbo1": num(grp.iloc[0]["PBO"]), "pbo2": num(grp.iloc[1]["PBO"]),
        "oos1": num(grp.iloc[0]["in-sample winner: IR out of sample"]), "oos2": num(grp.iloc[1]["in-sample winner: IR out of sample"]),
        "table_baseline": table_baseline(), "table_construction": table_construction(), "table_inference": table_inference(),
    }
    tex = TEX
    for k, v in fill.items():
        tex = tex.replace(f"<<{k}>>", v)
    missing = [part.split(">>")[0] for part in tex.split("<<")[1:]]
    assert not missing, f"unfilled: {missing}"
    (OUT / "research-note.tex").write_text(tex)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "research-note.tex"],
                           cwd=OUT, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    for ext in ("aux", "log", "out"):
        (OUT / f"research-note.{ext}").unlink(missing_ok=True)
    print("wrote", OUT / "research-note.pdf")


if __name__ == "__main__":
    build()
