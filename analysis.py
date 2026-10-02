"""TMD diagnoses and neck pain: complete analysis (single script).

Reads only data/tmd_cervical_master.csv (de-identified). Writes outputs/tables/*.csv, outputs/figures/*.png and
outputs/results*.json. Sections run in order; each keeps its own random seed, so results are reproducible.
  1. prespecified_and_multiverse: Firth primary/secondary models, power, ordinal and hurdle models, dose-response,
     palpation-vs-movement contrast, 64-specification multiverse with stratified permutation inference, MCA, LCA
  2. revision_analyses: 3-level exposure, crude prevalence, GEE contrast, adults-only and referral sensitivity,
     Greenacre-adjusted MCA inertia
  3. grouped_and_subgroups: painful TMD vs none with sequential adjustment, painful burden, TMD headache model,
     subgroup interactions (post hoc)
  4. revision_figures: burden figure, specification curves, causal diagram
Run:  python analysis.py        (about 40 min)     QUICK=1 python analysis.py   (smoke test)
"""
import warnings
warnings.filterwarnings('ignore')

# ---- Firth-penalized logistic regression (firthlogist) with a scikit-learn >= 1.6 compatibility shim.
# Validated against R logistf on the sex2 reference data (see check_firth()).
import firthlogist
from sklearn.utils.validation import validate_data


class Firth(firthlogist.FirthLogisticRegression):
    # ponytail: firthlogist calls the removed BaseEstimator._validate_data; drop this shim once upstream is fixed
    def _validate_data(self, *args, **kwargs):
        return validate_data(self, *args, **kwargs)


def check_firth():
    import numpy as np
    X, y, _ = firthlogist.load_sex2()
    m = Firth().fit(X, y)
    assert np.allclose(np.r_[m.coef_, m.intercept_], [-1.106, -0.0688, 2.2689, -2.1114, -0.7883, 3.096, 0.1203], atol=1e-3)
    assert np.allclose(m.ci_[0], [-1.9738, -0.3074], atol=1e-3)


def prespecified_and_multiverse():
    """Section from former analysis.py."""
    import json, warnings, itertools
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.metrics import roc_auc_score, adjusted_rand_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")
    HERE = Path(__file__).parent
    TAB, FIG = HERE / "outputs" / "tables", HERE / "outputs" / "figures"
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    SEED, N_PERM, N_BOOT, N_SIM, N_LCA_BOOT = 20261001, 2000, 2000, 2000, 200
    if __import__("os").environ.get("QUICK"):  # smoke test only
        N_PERM = N_BOOT = N_SIM = 20; N_LCA_BOOT = 5
    rng = np.random.default_rng(SEED)
    RES = {}

    # Okabe-Ito, square 300-dpi figures, one homogeneous typeface
    OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
          "sky": "#56B4E9", "pink": "#CC79A7", "yellow": "#F0E442", "black": "#000000"}
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "figure.figsize": (3.5, 3.5)})
    DXC = {"MUSCLE": OI["verm"], "JOINT": OI["blue"], "DISC": OI["green"]}
    DX = ["MUSCLE", "JOINT", "DISC"]


    # ----------------------------------------------------------------- derive (§2)
    def any_known(df, cols):
        """1 if any item == 1, 0 if all items == 0, NaN if undecidable (blank = not recorded)."""
        d = df[cols]
        return np.where((d == 1).any(axis=1), 1.0, np.where((d == 0).all(axis=1), 0.0, np.nan))


    m = pd.read_csv(HERE / "data" / "tmd_cervical_master.csv")
    MOV = ["mov_pain_flexext", "mov_pain_rot", "mov_pain_lat"]
    PALP = ["palp_trap_r", "palp_trap_l", "palp_subocc_r", "palp_subocc_l", "palp_scm_r", "palp_scm_l"]
    d = pd.DataFrame({"age": m.age.astype(float), "female": m.female.astype(float),
                      "hyper": m.hypermobility.astype(float)})
    d["MUSCLE"] = any_known(m, ["myalgia_local", "myofascial", "myofascial_referral"])
    d["MUSCLE_alt"] = m.clin_muscle_disorder
    d["JOINT"] = any_known(m, ["arthralgia", "arthritis"])
    d["JOINT_alt"] = m.arthralgia
    d["DISC"] = any_known(m, ["ddwr", "ddwr_locking", "ddwor_limited", "ddwor_unlimited"])
    d["DISC_alt"] = m.clin_joint_disorder
    mov = any_known(m, MOV)
    d["NP_move"] = mov
    d["NP_palp"] = m.palp_flag.astype(float)
    d["NP_clin"] = np.where((mov == 1) | (m.palp_flag == 1), 1.0, np.where((mov == 0) & (m.palp_flag == 0), 0.0, np.nan))
    d["NP_strict"] = np.where((mov == 1) & (m.palp_flag == 1), 1.0, np.where((mov == 0) | (m.palp_flag == 0), 0.0, np.nan))
    d["CPS"] = m[PALP].sum(axis=1)
    d["CPS_cat"] = pd.cut(d.CPS, [-1, 0, 10, 20, 30, 60], labels=False)
    d["MPC"] = m[MOV].sum(axis=1)
    d["SOURCES"] = d.MUSCLE + d.JOINT
    assert d[["NP_clin", "NP_move", "NP_strict", "NP_palp"]].sum().tolist() == [110, 60, 58, 108]
    assert d[DX].sum().tolist() == [115, 61, 64] and d.isna().sum().sum() == 0
    RES["n"] = len(d)
    RES["counts"] = {c: int(d[c].sum()) for c in ["NP_clin", "NP_move", "NP_strict", "NP_palp", *DX, "female", "hyper"]}

    # ----------------------------------------------------------------- helpers
    def firth(df, y, xs, full=True):
        X, Y = df[xs].to_numpy(float), df[y].to_numpy(int)
        return Firth().fit(X, Y) if full else Firth(wald=True, skip_ci=True).fit(X, Y)


    def flic(df, y, xs, f):
        """Firth coefficients + ML re-estimated intercept (Puhr et al. 2017) -> predicted probabilities."""
        eta = df[xs].to_numpy(float) @ f.coef_
        g = sm.GLM(df[y].to_numpy(int), np.ones((len(df), 1)), family=sm.families.Binomial(), offset=eta).fit()
        return 1 / (1 + np.exp(-(eta + g.params[0])))


    def evalue(or_, lo, hi):
        """E-value for point estimate and CI limit closest to 1; common outcome -> RR ≈ sqrt(OR)."""
        def e(rr):
            rr = 1 / rr if rr < 1 else rr
            return rr + np.sqrt(rr * (rr - 1))
        rr, rlo, rhi = np.sqrt(or_), np.sqrt(lo), np.sqrt(hi)
        ci = 1.0 if rlo <= 1 <= rhi else e(rlo if rr > 1 else rhi)
        return e(rr), ci


    def strata_perm(strata):
        idx = np.arange(len(strata))
        out = idx.copy()
        for s in np.unique(strata):
            w = idx[strata == s]
            out[w] = rng.permutation(w)
        return out


    COV = ["age", "female"]
    PRIM_X = DX + COV

    # ----------------------------------------------------------------- age form (§2.3, S6)
    def rcs3(x):
        k = np.percentile(x, [10, 50, 90])
        p = lambda u: np.clip(u, 0, None) ** 3
        return (p(x - k[0]) - p(x - k[1]) * (k[2] - k[0]) / (k[2] - k[1])
                + p(x - k[2]) * (k[1] - k[0]) / (k[2] - k[1])) / (k[2] - k[0]) ** 2


    d["age_nl"] = rcs3(d.age.to_numpy())
    lin = sm.Logit(d.NP_clin, sm.add_constant(d[PRIM_X])).fit(disp=0)
    spl = sm.Logit(d.NP_clin, sm.add_constant(d[PRIM_X + ["age_nl"]])).fit(disp=0)
    p_spline = float(1 - __import__("scipy").stats.chi2.cdf(2 * (spl.llf - lin.llf), 1))
    RES["age_spline_LRT_p"] = p_spline
    if p_spline < 0.05:
        COV = ["age", "age_nl", "female"]; PRIM_X = DX + COV

    # ----------------------------------------------------------------- §3 primary + key secondary
    rows, fits = [], {}
    for y in ["NP_clin", "NP_move"]:
        f = firth(d, y, PRIM_X); fits[y] = f
        ml = sm.Logit(d[y], sm.add_constant(d[PRIM_X])).fit(disp=0)
        for i, x in enumerate(PRIM_X):
            rows.append({"outcome": y, "term": x, "OR": np.exp(f.coef_[i]), "lo": np.exp(f.ci_[i, 0]),
                         "hi": np.exp(f.ci_[i, 1]), "p": f.pvals_[i], "OR_ML": np.exp(ml.params[x]),
                         "lo_ML": np.exp(ml.conf_int().loc[x, 0]), "hi_ML": np.exp(ml.conf_int().loc[x, 1]),
                         "p_ML": ml.pvalues[x]})
    main = pd.DataFrame(rows)
    isdx = main.term.isin(DX)
    main.loc[isdx, "p_holm6"] = multipletests(main.loc[isdx, "p"], method="holm")[1]
    for i in main.index[isdx & main.term.isin(["MUSCLE", "JOINT"])]:
        main.loc[i, ["E_point", "E_ci"]] = evalue(*main.loc[i, ["OR", "lo", "hi"]])
    main.to_csv(TAB / "T2_main_firth_NPclin_NPmove.csv", index=False)

    X_vif = sm.add_constant(d[PRIM_X]).to_numpy(float)
    RES["VIF"] = {x: float(variance_inflation_factor(X_vif, i + 1)) for i, x in enumerate(PRIM_X)}
    RES["discrimination"] = {}
    for y in ["NP_clin", "NP_move"]:
        p = flic(d, y, PRIM_X, fits[y]); Y = d[y].to_numpy(int)
        boots = [roc_auc_score(Y[b], p[b]) for b in (rng.integers(0, len(Y), len(Y)) for _ in range(N_BOOT))
                 if Y[b].min() != Y[b].max()]
        lp = np.log(p / (1 - p)); cal = sm.Logit(Y, sm.add_constant(lp)).fit(disp=0)
        RES["discrimination"][y] = {"AUC": roc_auc_score(Y, p), "AUC_lo": np.percentile(boots, 2.5),
                                    "AUC_hi": np.percentile(boots, 97.5), "cal_intercept": float(cal.params[0]),
                                    "cal_slope": float(cal.params[1])}

    # ----------------------------------------------------------------- §4 power (resampled design)
    def sim_power(y, dx, ors):
        null = sm.Logit(d[y], sm.add_constant(d[COV])).fit(disp=0)
        out = []
        for OR in ors:
            hit = 0
            for _ in range(N_SIM):
                b = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
                lp = null.params["const"] + b[COV].to_numpy() @ null.params[COV].to_numpy() + np.log(OR) * b[dx].to_numpy()
                b["ysim"] = (rng.random(len(b)) < 1 / (1 + np.exp(-lp))).astype(int)
                if b.ysim.min() == b.ysim.max():
                    continue
                f = firth(b, "ysim", PRIM_X, full=False)
                hit += f.pvals_[PRIM_X.index(dx)] < 0.05
            out.append(hit / N_SIM)
        return out


    ORS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    pw = pd.DataFrame([{"outcome": y, "dx": dx, "OR": o, "power": pv}
                       for y in ["NP_clin", "NP_move"] for dx in DX for o, pv in zip(ORS, sim_power(y, dx, ORS))])
    pw.to_csv(TAB / "S_power_simulation.csv", index=False)
    RES["MDOR80"] = {f"{y}|{dx}": (float(np.interp(0.8, g.power, g.OR)) if g.power.max() >= 0.8 else ">4")
                     for (y, dx), g in pw.groupby(["outcome", "dx"])}

    # ----------------------------------------------------------------- §5 ordinal secondaries
    ord_rows, po_rows = [], []
    for y in ["CPS_cat", "MPC"]:
        om = OrderedModel(d[y].astype(int), d[PRIM_X], distr="logit").fit(method="bfgs", disp=0)
        ci = om.conf_int()
        for x in PRIM_X:
            ord_rows.append({"outcome": y, "term": x, "OR": np.exp(om.params[x]), "lo": np.exp(ci.loc[x, 0]),
                             "hi": np.exp(ci.loc[x, 1]), "p": om.pvalues[x]})
        cuts = sorted(d[y].unique())[1:]
        for c in cuts:
            b = sm.Logit((d[y] >= c).astype(int), sm.add_constant(d[PRIM_X])).fit(disp=0)
            po_rows.append({"outcome": y, "cut": f">={c}", **{f"OR_{x}": np.exp(b.params[x]) for x in DX}})
    pd.DataFrame(ord_rows).to_csv(TAB / "T3_ordinal_CPS_MPC.csv", index=False)
    po = pd.DataFrame(po_rows); po.to_csv(TAB / "S_proportional_odds_check.csv", index=False)
    RES["PO_flags"] = {f"{y}|{x}": bool(max(g[f"OR_{x}"].iloc[[0, -1]]) / min(g[f"OR_{x}"].iloc[[0, -1]]) > 2)
                       for y, g in po.groupby("outcome") for x in DX}
    # hurdle sensitivity for CPS
    d["CPS_pos"] = (d.CPS > 0).astype(int)
    h1 = firth(d, "CPS_pos", PRIM_X)
    pos = d[d.CPS > 0]
    h2 = sm.OLS(np.log(pos.CPS), sm.add_constant(pos[PRIM_X])).fit()
    pd.DataFrame([{"part": "CPS>0 (Firth OR)", "term": x, "est": np.exp(h1.coef_[i]), "lo": np.exp(h1.ci_[i, 0]),
                   "hi": np.exp(h1.ci_[i, 1]), "p": h1.pvals_[i]} for i, x in enumerate(PRIM_X)] +
                 [{"part": "log CPS | CPS>0 (ratio of geometric means)", "term": x, "est": np.exp(h2.params[x]),
                   "lo": np.exp(h2.conf_int().loc[x, 0]), "hi": np.exp(h2.conf_int().loc[x, 1]), "p": h2.pvalues[x]}
                  for x in PRIM_X]).to_csv(TAB / "S_hurdle_CPS.csv", index=False)

    # ----------------------------------------------------------------- §6 dose-response
    dr = []
    for y in ["NP_clin", "NP_move"]:
        xs = ["SOURCES", "DISC"] + COV
        f = firth(d, y, xs)
        dr.append({"outcome": y, "OR_per_source": np.exp(f.coef_[0]), "lo": np.exp(f.ci_[0, 0]),
                   "hi": np.exp(f.ci_[0, 1]), "p": f.pvals_[0]})
        dr[-1].update({f"prev_{k}_sources": float(d.loc[d.SOURCES == k, y].mean()) for k in (0, 1, 2)})
    pd.DataFrame(dr).to_csv(TAB / "T4_dose_response.csv", index=False)

    # ----------------------------------------------------------------- §8 multiverse (+ §7 permutation p)
    SPECS = list(itertools.product(["NP_clin", "NP_move", "NP_strict", "NP_palp"],
                                   ["MUSCLE", "MUSCLE_alt"], ["JOINT", "JOINT_alt"], ["DISC", "DISC_alt"],
                                   ["base", "hyper"]))
    EXPO = ["MUSCLE", "MUSCLE_alt", "JOINT", "JOINT_alt", "DISC", "DISC_alt"]


    def run_specs(df):
        """log-OR and Wald p for each diagnosis in each of the 64 specifications."""
        b = np.empty((len(SPECS), 3)); p = np.empty((len(SPECS), 3))
        for s, (y, mu, jo, di, cv) in enumerate(SPECS):
            xs = [mu, jo, di] + COV + (["hyper"] if cv == "hyper" else [])
            f = firth(df, y, xs, full=False)
            b[s], p[s] = f.coef_[:3], f.pvals_[:3]
        return b, p


    def spec_stats(b, p):
        med = np.median(b, axis=0)
        sign = np.sign(med)
        nsig = ((p < 0.05) & (np.sign(b) == sign)).sum(axis=0)
        return med, nsig


    def hdef(b):
        """Δ log-OR palpation - movement, primary groupings, base covariates."""
        i_p = SPECS.index(("NP_palp", "MUSCLE", "JOINT", "DISC", "base"))
        i_m = SPECS.index(("NP_move", "MUSCLE", "JOINT", "DISC", "base"))
        return b[i_p] - b[i_m]


    b_obs, p_obs = run_specs(d)
    med_obs, nsig_obs = spec_stats(b_obs, p_obs)
    hdef_obs = hdef(b_obs)
    strata = pd.qcut(d.age, 3, labels=False).to_numpy() * 2 + d.female.to_numpy().astype(int)
    null_med, null_nsig, null_hdef = [], [], []
    for _ in range(N_PERM):
        dp = d.copy()
        dp[EXPO] = d[EXPO].to_numpy()[strata_perm(strata)]
        b, p = run_specs(dp)
        md, ns = spec_stats(b, p)
        null_med.append(md); null_nsig.append(ns); null_hdef.append(hdef(b))
    null_med, null_nsig, null_hdef = map(np.array, (null_med, null_nsig, null_hdef))
    p_med = ((np.abs(null_med) >= np.abs(med_obs)).sum(0) + 1) / (N_PERM + 1)
    p_nsig = ((null_nsig >= nsig_obs).sum(0) + 1) / (N_PERM + 1)
    p_hdef = ((np.abs(null_hdef) >= np.abs(hdef_obs)).sum(0) + 1) / (N_PERM + 1)
    mv = pd.DataFrame([{"outcome": s[0], "muscle_coding": s[1], "joint_coding": s[2], "disc_coding": s[3],
                        "covariates": s[4], **{f"OR_{x}": np.exp(b_obs[k, j]) for j, x in enumerate(DX)},
                        **{f"p_{x}": p_obs[k, j] for j, x in enumerate(DX)}} for k, s in enumerate(SPECS)])
    mv.to_csv(TAB / "S_multiverse_64_specifications.csv", index=False)
    RES["multiverse"] = {x: {"median_OR": float(np.exp(med_obs[j])), "n_sig_dominant": int(nsig_obs[j]),
                             "p_median": float(p_med[j]), "p_nsig": float(p_nsig[j])} for j, x in enumerate(DX)}
    for key, arr in [("p_median_holm", p_med), ("p_nsig_holm", p_nsig)]:
        for j, x in enumerate(DX):
            RES["multiverse"][x][key] = float(multipletests(arr, method="holm")[1][j])

    # ----------------------------------------------------------------- §7 H-def bootstrap CIs
    boot_bin, boot_ord = [], []
    for _ in range(N_BOOT):
        bi = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
        fp, fm = firth(bi, "NP_palp", PRIM_X, False), firth(bi, "NP_move", PRIM_X, False)
        boot_bin.append(fp.coef_[:3] - fm.coef_[:3])
        try:
            oc = OrderedModel(bi.CPS_cat.astype(int), bi[PRIM_X], distr="logit").fit(method="bfgs", disp=0).params
            om_ = OrderedModel(bi.MPC.astype(int), bi[PRIM_X], distr="logit").fit(method="bfgs", disp=0).params
            boot_ord.append(oc[DX].to_numpy() - om_[DX].to_numpy())
        except Exception:
            pass
    boot_bin, boot_ord = np.array(boot_bin), np.array(boot_ord)
    o_c = pd.DataFrame(ord_rows).set_index(["outcome", "term"])
    ord_delta = np.log(o_c.loc["CPS_cat"].loc[DX, "OR"].to_numpy()) - np.log(o_c.loc["MPC"].loc[DX, "OR"].to_numpy())
    hd = pd.DataFrame([{"contrast": "binary: NP_palp - NP_move", "dx": x, "delta_logOR": hdef_obs[j],
                        "ratio_of_OR": np.exp(hdef_obs[j]), "lo": np.exp(np.percentile(boot_bin[:, j], 2.5)),
                        "hi": np.exp(np.percentile(boot_bin[:, j], 97.5)), "p_perm": p_hdef[j]} for j, x in enumerate(DX)] +
                      [{"contrast": "ordinal: CPS_cat - MPC", "dx": x, "delta_logOR": ord_delta[j],
                        "ratio_of_OR": np.exp(ord_delta[j]), "lo": np.exp(np.percentile(boot_ord[:, j], 2.5)),
                        "hi": np.exp(np.percentile(boot_ord[:, j], 97.5)), "p_perm": np.nan} for j, x in enumerate(DX)])
    hd.to_csv(TAB / "T5_Hdef_palpation_vs_movement.csv", index=False)

    # ----------------------------------------------------------------- §10a MCA
    import prince
    IND = pd.DataFrame({"Myalgia": d.MUSCLE, "Joint pain": d.JOINT, "DDwR": m.ddwr,
                        "DD locking/woR": any_known(m, ["ddwr_locking", "ddwor_limited", "ddwor_unlimited"]),
                        "Subluxation": m.subluxation, "DJD": any_known(m, ["osteoarthrosis", "osteoarthritis"]),
                        "Contracture/hypertrophy": any_known(m, ["contracture", "hypertrophy"])})
    cc = IND.notna().all(axis=1).to_numpy()   # complete cases (blank = not recorded, SAP DC4)
    RES["MCA_LCA_n"] = int(cc.sum())
    IND = IND[cc].astype(int); dm = d[cc].reset_index(drop=True); mm = m[cc].reset_index(drop=True)
    assert (IND.sum() >= 15).all()
    mca = prince.MCA(n_components=4, random_state=SEED).fit(IND.astype(int).astype(str))
    Q = IND.shape[1]
    eig = np.asarray(mca.eigenvalues_)
    benz = np.where(eig > 1 / Q, (Q / (Q - 1)) ** 2 * (eig - 1 / Q) ** 2, 0)
    RES["MCA_benzecri_pct"] = (100 * benz / benz.sum()).round(1).tolist()
    contrib = mca.column_contributions_.iloc[:, :2]
    contrib.to_csv(TAB / "S_MCA_contributions.csv")
    coords = mca.row_coordinates(IND.astype(int).astype(str)).iloc[:, :2].to_numpy()
    dm["MCA1"], dm["MCA2"] = coords[:, 0], coords[:, 1]
    mca_fit = []
    for y in ["NP_clin", "NP_move"]:
        f = firth(dm, y, ["MCA1", "MCA2"] + COV)
        mca_fit += [{"outcome": y, "term": t_, "OR_per_unit": np.exp(f.coef_[i]), "lo": np.exp(f.ci_[i, 0]),
                     "hi": np.exp(f.ci_[i, 1]), "p": f.pvals_[i]} for i, t_ in enumerate(["MCA1", "MCA2"])]
    pd.DataFrame(mca_fit).to_csv(TAB / "S_MCA_dimensions_vs_neckpain.csv", index=False)
    cat_xy = mca.column_coordinates(IND.astype(int).astype(str)).iloc[:, :2]

    # ----------------------------------------------------------------- §10b LCA (exploratory)
    from stepmix.stepmix import StepMix
    Xl = IND.to_numpy(int)


    def lca(k, X, n_init=50, seed=SEED):
        return StepMix(n_components=k, measurement="binary", n_init=n_init, random_state=seed, verbose=0, progress_bar=0).fit(X)


    lca_rows, models = [], {}
    for k in range(1, 5):
        mdl = lca(k, Xl); models[k] = mdl
        post = mdl.predict_proba(Xl)
        ent = 1.0 if k == 1 else 1 - (-(post * np.log(np.clip(post, 1e-12, 1))).sum()) / (len(Xl) * np.log(k))
        share = np.bincount(post.argmax(1), minlength=k) / len(Xl)
        lca_rows.append({"k": k, "BIC": mdl.bic(Xl), "entropy": ent, "smallest_class": share.min(),
                         "admissible": k == 1 or (ent >= 0.70 and share.min() >= 0.10)})
    lt = pd.DataFrame(lca_rows); lt.to_csv(TAB / "S_LCA_fit.csv", index=False)
    k_best = int(lt[lt.admissible].sort_values("BIC").k.iloc[0])
    RES["LCA"] = {"k_selected": k_best}
    if k_best > 1:
        best = models[k_best]; lab = best.predict(Xl)
        par = best.get_parameters()
        w, pis = par["weights"], par["measurement"]["pis"]
        bvr = {}
        for i, j in itertools.combinations(range(Q), 2):
            E = np.array([[len(Xl) * (w * (pis[:, i] if a else 1 - pis[:, i]) * (pis[:, j] if b_ else 1 - pis[:, j])).sum()
                           for b_ in (1, 0)] for a in (1, 0)])
            O = np.array([[((Xl[:, i] == a) & (Xl[:, j] == b_)).sum() for b_ in (1, 0)] for a in (1, 0)])
            bvr[f"{IND.columns[i]}|{IND.columns[j]}"] = float(((O - E) ** 2 / E).sum())
        aris = []
        for s in range(N_LCA_BOOT):
            bi = rng.integers(0, len(Xl), len(Xl))
            aris.append(adjusted_rand_score(lab, lca(k_best, Xl[bi], n_init=10, seed=s).predict(Xl)))
        prof = IND.assign(cls=lab, NP_clin=dm.NP_clin, NP_move=dm.NP_move, headache=mm.headache_tmd).groupby("cls").mean()
        prof["n"] = np.bincount(lab); prof.to_csv(TAB / "S_LCA_class_profiles.csv")
        bch = {}
        for y in ["NP_clin", "NP_move"]:
            s3 = StepMix(n_components=k_best, measurement="binary", structural="binary", n_steps=3, correction="BCH",
                         n_init=50, random_state=SEED, verbose=0, progress_bar=0).fit(Xl, dm[[y]].to_numpy(int))
            bch[y] = np.asarray(s3.get_parameters()["structural"]["pis"]).ravel().round(3).tolist()
        RES["LCA"].update({"max_BVR": max(bvr.values()), "n_BVR_gt_3.84": int(sum(v > 3.84 for v in bvr.values())),
                           "ARI_median": float(np.median(aris)), "ARI_IQR": np.percentile(aris, [25, 75]).tolist(),
                           "BCH_outcome_prob_by_class": bch})

    # ----------------------------------------------------------------- figures
    def save(fig, name):
        for ax in fig.axes:  # plain tick labels on log axes (0.5, 1, 2, 4 instead of 10^0)
            for a, sc in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
                if sc == "log":
                    a.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
                    a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
                    a.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.tight_layout(); fig.savefig(FIG / name); plt.close(fig)


    # F1 forest: primary vs key secondary
    fig, ax = plt.subplots()
    for k, y in enumerate(["NP_clin", "NP_move"]):
        r = main[(main.outcome == y) & main.term.isin(DX)]
        ys = np.arange(3) + (0.15 if k == 0 else -0.15)
        ax.errorbar(r.OR, ys, xerr=[r.OR - r.lo, r.hi - r.OR], fmt="o" if k == 0 else "s", ms=4, capsize=2,
                    color=OI["black"] if k == 0 else OI["sky"],
                    label="Movement or palpation (primary)" if k == 0 else "Movement pain (key secondary)")
    ax.axvline(1, color="grey", lw=0.8, ls="--"); ax.set_xscale("log")
    ax.set_yticks(range(3)); ax.set_yticklabels(["Myalgia", "Joint pain", "Disc displacement"]); ax.invert_yaxis()
    ax.set_xlabel("Adjusted odds ratio (Firth, 95% CI)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("TMD diagnoses and neck pain", fontsize=9)
    save(fig, "F1_forest_TMD_diagnoses_neck_pain_primary_vs_movement.png")

    # F2 specification curves (one per diagnosis)
    OUTC = {"NP_clin": OI["black"], "NP_move": OI["sky"], "NP_strict": OI["blue"], "NP_palp": OI["orange"]}
    for j, x in enumerate(DX):
        order = np.argsort(b_obs[:, j])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.3]})
        for r, s in enumerate(order):
            y = SPECS[s][0]
            prim = SPECS[s] == ("NP_clin", "MUSCLE", "JOINT", "DISC", "base")
            a1.scatter(r, np.exp(b_obs[s, j]), s=22 if prim else 9, color=OUTC[y],
                       marker="D" if prim else "o", edgecolor="k" if prim else "none", zorder=3)
            if p_obs[s, j] < 0.05:
                a1.scatter(r, np.exp(b_obs[s, j]), s=40, facecolor="none", edgecolor=OI["verm"], lw=0.7)
        a1.axhline(1, color="grey", lw=0.8, ls="--"); a1.set_yscale("log"); a1.set_ylabel("Adjusted OR")
        a1.set_title(f"{['Myalgia', 'Joint pain', 'Disc displacement'][j]}: 64 specifications\n"
                     f"median OR {np.exp(med_obs[j]):.2f}; joint permutation p = {p_med[j]:.3f}", fontsize=8)
        rowsl = [("NP_clin", 0), ("NP_move", 0), ("NP_strict", 0), ("NP_palp", 0),
                 ("alt grouping", 1), ("+ hypermobility", 4)]
        for rr, (lab_, pos_) in enumerate(rowsl):
            for r, s in enumerate(order):
                sp = SPECS[s]
                on = (sp[0] == lab_) if pos_ == 0 else (sp[1 + j].endswith("_alt") if pos_ == 1 else sp[4] == "hyper")
                if on:
                    a2.scatter(r, rr, s=5, marker="s", color=OUTC.get(lab_, OI["black"]))
        a2.set_yticks(range(len(rowsl))); a2.set_yticklabels([r_[0] for r_ in rowsl], fontsize=7)
        a2.invert_yaxis(); a2.set_xlabel("Specification (ranked by OR)"); a2.set_xticks([])
        save(fig, f"F2_specification_curve_{x.lower()}_64_definitions.png")

    # F3 H-def
    fig, ax = plt.subplots()
    for k, cname in enumerate(["binary: NP_palp - NP_move", "ordinal: CPS_cat - MPC"]):
        r = hd[hd.contrast == cname]
        ys = np.arange(3) + (0.15 if k == 0 else -0.15)
        ax.errorbar(r.ratio_of_OR, ys, xerr=[r.ratio_of_OR - r.lo, r.hi - r.ratio_of_OR], fmt="o" if k == 0 else "s",
                    ms=4, capsize=2, color=OI["orange"] if k == 0 else OI["pink"],
                    label="Palpation vs movement (binary)" if k == 0 else "Palpation score vs movement count")
    ax.axvline(1, color="grey", lw=0.8, ls="--"); ax.set_xscale("log")
    ax.set_yticks(range(3)); ax.set_yticklabels(["Myalgia", "Joint pain", "Disc displacement"]); ax.invert_yaxis()
    ax.set_xlabel("Ratio of ORs (palpation / movement)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Effect of the neck-pain definition", fontsize=9)
    save(fig, "F3_definition_effect_palpation_vs_movement_ratio_of_OR.png")

    # F4 MCA biplot
    fig, ax = plt.subplots()
    for v, c, lab_ in [(1, OI["black"], "Neck pain (primary)"), (0, OI["sky"], "No neck pain")]:
        w_ = dm.NP_clin == v
        ax.scatter(dm.MCA1[w_] + rng.normal(0, 0.02, w_.sum()), dm.MCA2[w_] + rng.normal(0, 0.02, w_.sum()),
                   s=8, alpha=0.6, color=c, label=lab_)
    for name in IND.columns:
        key = [i for i in cat_xy.index if str(i).startswith(name) and str(i).endswith("1")]
        if key:
            xy = cat_xy.loc[key[0]]
            ax.scatter(*xy, marker="^", color=OI["verm"], s=25); ax.annotate(name, xy, fontsize=6.5, xytext=(3, 3),
                                                                            textcoords="offset points")
    ax.set_xlabel(f"Dimension 1 ({RES['MCA_benzecri_pct'][0]}% adj. inertia)")
    ax.set_ylabel(f"Dimension 2 ({RES['MCA_benzecri_pct'][1]}% adj. inertia)")
    ax.legend(frameon=False, fontsize=7); ax.set_title("DC/TMD diagnostic profile (MCA)", fontsize=9)
    save(fig, "F4_MCA_DCTMD_profile_by_neck_pain.png")

    # F5 power curves
    fig, ax = plt.subplots()
    for (y, dx), g in pw.groupby(["outcome", "dx"]):
        ax.plot(g.OR, g.power, marker="o", ms=3, color=DXC[dx], ls="-" if y == "NP_clin" else "--",
                label=f"{dx.title()} — {'primary' if y == 'NP_clin' else 'movement'}")
    ax.axhline(0.8, color="grey", lw=0.8, ls=":"); ax.set_ylim(0, 1)
    ax.set_xlabel("True adjusted OR"); ax.set_ylabel("Power (α = 0.05, n = 145)")
    ax.legend(frameon=False, fontsize=6); ax.set_title("Detectable effect sizes", fontsize=9)
    save(fig, "F5_power_curves_minimum_detectable_OR.png")

    # ----------------------------------------------------------------- dump
    def clean(o):
        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o


    RES["main_table"] = main.round(4).to_dict("records")
    RES["Hdef"] = hd.round(4).to_dict("records")
    RES["dose_response"] = pd.DataFrame(dr).round(4).to_dict("records")
    (HERE / "outputs" / "results.json").write_text(json.dumps(clean(RES), indent=1, default=str), encoding="utf-8")
    print(json.dumps(clean({k: RES[k] for k in ["counts", "age_spline_LRT_p", "VIF", "multiverse", "LCA", "MDOR80"]}),
                     indent=1, default=str))
    print(main.round(3).to_string(index=False))
    print(hd.round(3).to_string(index=False))


def revision_analyses():
    """Section from former analysis_revision.py."""
    import json, warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    warnings.filterwarnings("ignore")
    HERE = Path(__file__).parent
    TAB = HERE / "outputs" / "tables"
    rng = np.random.default_rng(20261001)
    N_BOOT = 2000
    R = {}


    def any_known(df, cols):
        d = df[cols]
        return np.where((d == 1).any(axis=1), 1.0, np.where((d == 0).all(axis=1), 0.0, np.nan))


    # ponytail: duplicates the derive block of analysis.py (which runs on import); keep both in sync
    m = pd.read_csv(HERE / "data" / "tmd_cervical_master.csv")
    MOV = ["mov_pain_flexext", "mov_pain_rot", "mov_pain_lat"]
    PALP = ["palp_trap_r", "palp_trap_l", "palp_subocc_r", "palp_subocc_l", "palp_scm_r", "palp_scm_l"]
    d = pd.DataFrame({"age": m.age.astype(float), "female": m.female.astype(float)})
    d["MUSCLE"] = any_known(m, ["myalgia_local", "myofascial", "myofascial_referral"])
    d["JOINT"] = any_known(m, ["arthralgia", "arthritis"])
    d["DISC"] = any_known(m, ["ddwr", "ddwr_locking", "ddwor_limited", "ddwor_unlimited"])
    mov = any_known(m, MOV)
    d["NP_move"] = mov
    d["NP_palp"] = m.palp_flag.astype(float)
    d["NP_clin"] = np.where((mov == 1) | (m.palp_flag == 1), 1.0, 0.0)
    d["CPS_cat"] = pd.cut(m[PALP].sum(axis=1), [-1, 0, 10, 20, 30, 60], labels=False)
    d["MPC"] = m[MOV].sum(axis=1)
    d["referral"] = m.myofascial_referral.fillna(0)
    assert d[["NP_clin", "NP_move", "NP_palp"]].sum().tolist() == [110, 60, 108]

    # ---- R6 nesting + 3-level exposure
    xt = pd.crosstab(d.MUSCLE, d.JOINT)
    R["nesting_crosstab_muscle_by_joint"] = {f"M{int(i)}J{int(j)}": int(xt.loc[i, j]) for i in xt.index for j in xt.columns}
    assert ((d.JOINT == 1) & (d.MUSCLE == 0)).sum() == 0
    d["L_myo"] = ((d.MUSCLE == 1) & (d.JOINT == 0)).astype(float)   # myalgia only vs none
    d["L_both"] = ((d.MUSCLE == 1) & (d.JOINT == 1)).astype(float)  # myalgia + joint vs none
    LVL = ["L_myo", "L_both"]
    X3 = LVL + ["DISC", "age", "female"]


    def firth_rows(df, y, xs, tag):
        f = Firth().fit(df[xs].to_numpy(float), df[y].to_numpy(int))
        return [{"model": tag, "outcome": y, "term": x, "OR": np.exp(f.coef_[i]), "lo": np.exp(f.ci_[i, 0]),
                 "hi": np.exp(f.ci_[i, 1]), "p": f.pvals_[i]} for i, x in enumerate(xs)]


    rows = []
    for y in ["NP_clin", "NP_move"]:
        rows += firth_rows(d, y, X3, "3-level")
        # myalgia+joint vs myalgia only (the incremental joint contrast) = refit with myalgia-only as reference
        dd = d[d.MUSCLE == 1]
        rows += firth_rows(dd, y, ["JOINT", "DISC", "age", "female"], "within myalgia (joint vs none)")
    for y in ["CPS_cat", "MPC"]:
        om = OrderedModel(d[y].astype(int), d[X3], distr="logit").fit(method="bfgs", disp=0)
        c = om.conf_int()
        rows += [{"model": "3-level ordinal", "outcome": y, "term": x, "OR": np.exp(om.params[x]), "lo": np.exp(c.loc[x, 0]),
                  "hi": np.exp(c.loc[x, 1]), "p": om.pvalues[x]} for x in X3]
    # global 2-df test of the 3-level exposure (ML LRT), for comparison with the 1-df dose-response
    for y in ["NP_clin", "NP_move"]:
        full = sm.Logit(d[y], sm.add_constant(d[X3])).fit(disp=0)
        red = sm.Logit(d[y], sm.add_constant(d[["DISC", "age", "female"]])).fit(disp=0)
        from scipy.stats import chi2
        R[f"LRT_2df_exposure_{y}"] = float(1 - chi2.cdf(2 * (full.llf - red.llf), 2))
    pd.DataFrame(rows).to_csv(TAB / "R_3level_exposure.csv", index=False)

    # ---- R7 crude prevalence
    grp = np.select([d.L_both == 1, d.L_myo == 1], ["myalgia + joint pain", "myalgia only"], "neither")
    cr = []
    for y in ["NP_clin", "NP_move", "NP_palp"]:
        for g in ["neither", "myalgia only", "myalgia + joint pain"]:
            w = grp == g
            cr.append({"outcome": y, "exposure": g, "n": int(w.sum()), "events": int(d.loc[w, y].sum()),
                       "pct": 100 * d.loc[w, y].mean()})
        for g, w in [("disc displacement", d.DISC == 1), ("no disc displacement", d.DISC == 0)]:
            cr.append({"outcome": y, "exposure": g, "n": int(w.sum()), "events": int(d.loc[w, y].sum()),
                       "pct": 100 * d.loc[w, y].mean()})
    pd.DataFrame(cr).to_csv(TAB / "R_crude_prevalence.csv", index=False)

    # ---- R8 palpation vs movement: GEE on stacked outcomes + paired bootstrap p
    long = pd.concat([d.assign(y=d.NP_palp, palp=1, pid=d.index), d.assign(y=d.NP_move, palp=0, pid=d.index)])
    gee_rows = []
    for expo, terms in [("3 diagnoses", ["MUSCLE", "JOINT", "DISC"]), ("3-level", ["L_myo", "L_both", "DISC"])]:
        rhs = " + ".join(f"palp*{t}" for t in terms + ["age", "female"])
        g = smf.gee(f"y ~ {rhs}", "pid", long, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        for t in terms:
            k = f"palp:{t}"
            lo, hi = g.conf_int().loc[k]
            gee_rows.append({"exposure_coding": expo, "term": t, "ratio_of_OR": np.exp(g.params[k]), "lo": np.exp(lo),
                             "hi": np.exp(hi), "p_GEE": g.pvalues[k]})


    def delta(df, xs):
        a = Firth(wald=True, skip_ci=True).fit(df[xs].to_numpy(float), df.NP_palp.to_numpy(int)).coef_
        b = Firth(wald=True, skip_ci=True).fit(df[xs].to_numpy(float), df.NP_move.to_numpy(int)).coef_
        return a[:3] - b[:3]


    for expo, xs in [("3 diagnoses", ["MUSCLE", "JOINT", "DISC", "age", "female"]), ("3-level", X3)]:
        obs = delta(d, xs)
        bs = np.array([delta(d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True), xs) for _ in range(N_BOOT)])
        pboot = np.minimum(1, 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0)))
        for j in range(3):
            r = next(r for r in gee_rows if r["exposure_coding"] == expo and r["term"] == xs[j])
            r.update({"ratio_firth_obs": np.exp(obs[j]), "boot_lo": np.exp(np.percentile(bs[:, j], 2.5)),
                      "boot_hi": np.exp(np.percentile(bs[:, j], 97.5)), "p_boot": pboot[j]})
    pd.DataFrame(gee_rows).to_csv(TAB / "R_palpation_vs_movement_tests.csv", index=False)

    # ---- R9 adults only, R10 myalgia without referral
    sens = []
    ad = d[d.age >= 18].reset_index(drop=True)
    R["n_adults"] = len(ad); R["n_minors"] = int((d.age < 18).sum())
    for y in ["NP_clin", "NP_move"]:
        sens += firth_rows(ad, y, ["MUSCLE", "JOINT", "DISC", "age", "female"], "adults >=18")
        sens += firth_rows(ad, y, X3, "adults >=18, 3-level")
    nr = d[d.referral == 0].reset_index(drop=True)
    R["n_without_referral"] = len(nr)
    for y in ["NP_clin", "NP_move"]:
        sens += firth_rows(nr, y, ["MUSCLE", "JOINT", "DISC", "age", "female"], "excluding myofascial pain with referral")
    pd.DataFrame(sens).to_csv(TAB / "R_sensitivity_adults_referral.csv", index=False)

    # ---- R11 MCA inertia: Benzecri and Greenacre adjusted (all eigenvalues from the indicator matrix)
    IND = pd.DataFrame({"Myalgia": d.MUSCLE, "Joint pain": d.JOINT, "DDwR": m.ddwr,
                        "DD locking/woR": any_known(m, ["ddwr_locking", "ddwor_limited", "ddwor_unlimited"]),
                        "Subluxation": m.subluxation, "DJD": any_known(m, ["osteoarthrosis", "osteoarthritis"]),
                        "Contracture/hypertrophy": any_known(m, ["contracture", "hypertrophy"])}).dropna().astype(int)
    Z = pd.get_dummies(IND.astype(str)).to_numpy(float)
    P = Z / Z.sum(); r_, c_ = P.sum(1), P.sum(0)
    S = (P - np.outer(r_, c_)) / np.sqrt(np.outer(r_, c_))
    lam = np.linalg.svd(S, compute_uv=False) ** 2           # principal inertias
    Q, J = IND.shape[1], Z.shape[1]
    adj = np.where(lam > 1 / Q, (Q / (Q - 1)) ** 2 * (lam - 1 / Q) ** 2, 0)
    greenacre_total = Q / (Q - 1) * ((lam ** 2).sum() - (J - Q) / Q ** 2)
    R["MCA_inertia"] = {"benzecri_pct": (100 * adj[:3] / adj.sum()).round(1).tolist(),
                        "greenacre_pct": (100 * adj[:3] / greenacre_total).round(1).tolist(),
                        "n": len(IND)}

    # ---- R12 ordinal bootstrap failures (same scheme as analysis.py §7)
    fails = 0
    for _ in range(N_BOOT):
        bi = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
        try:
            for y in ["CPS_cat", "MPC"]:
                OrderedModel(bi[y].astype(int), bi[["MUSCLE", "JOINT", "DISC", "age", "female"]], distr="logit").fit(method="bfgs", disp=0)
        except Exception:
            fails += 1
    R["ordinal_bootstrap_failures"] = fails

    (HERE / "outputs" / "results_revision.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(json.dumps(R, indent=1))
    for f in ["R_3level_exposure", "R_crude_prevalence", "R_palpation_vs_movement_tests", "R_sensitivity_adults_referral"]:
        print(f"\n== {f}"); print(pd.read_csv(TAB / f"{f}.csv").round(3).to_string(index=False))


def grouped_and_subgroups():
    """Section from former analysis_subgroups.py."""
    import json, warnings
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.stats.multitest import multipletests

    warnings.filterwarnings("ignore")
    HERE = Path(__file__).parent
    TAB, FIG = HERE / "outputs" / "tables", HERE / "outputs" / "figures"
    m = pd.read_csv(HERE / "data" / "tmd_cervical_master.csv")
    anyof = lambda cols: (m[cols] == 1).any(axis=1).astype(float)
    mov = anyof(["mov_pain_flexext", "mov_pain_rot", "mov_pain_lat"])
    d = pd.DataFrame({
        "age": m.age.astype(float), "female": m.female.astype(float), "hyper": m.hypermobility.astype(float),
        "MUSCLE": anyof(["myalgia_local", "myofascial", "myofascial_referral"]), "JOINT": anyof(["arthralgia", "arthritis"]),
        "DISC": anyof(["ddwr", "ddwr_locking", "ddwor_limited", "ddwor_unlimited"]), "HEAD": m.headache_tmd.astype(float),
        "NP_clin": ((mov == 1) | (m.palp_flag == 1)).astype(float), "NP_move": mov})
    assert d[["NP_clin", "NP_move", "MUSCLE", "JOINT", "DISC"]].sum().tolist() == [110, 60, 115, 61, 64]
    d["PAINFUL"] = ((d.MUSCLE == 1) | (d.JOINT == 1)).astype(float)          # any painful TMD diagnosis
    d["BURDEN"] = d.MUSCLE + d.JOINT + d.HEAD                                  # painful TMD diagnoses incl. TMD headache (0-3)
    OUTS = ["NP_clin", "NP_move"]
    LAB = {"NP_clin": "Clinical definition", "NP_move": "Movement pain"}
    RES = {"n_painful": int(d.PAINFUL.sum()), "headache_all_painful": bool(((d.HEAD == 1) & (d.PAINFUL == 0)).sum() == 0)}


    def firth(df, y, xs):
        f = Firth().fit(df[xs].to_numpy(float), df[y].to_numpy(int))
        return {x: (np.exp(f.coef_[i]), np.exp(f.ci_[i, 0]), np.exp(f.ci_[i, 1]), f.pvals_[i]) for i, x in enumerate(xs)}


    # ---- 1. grouped exposures: crude prevalence and adjusted OR
    grp = []
    for y in OUTS:
        for expo, levels in [("PAINFUL", [0, 1]), ("BURDEN", [0, 1, 2, 3])]:
            for v in levels:
                w = d[expo] == v
                grp.append({"outcome": y, "exposure": expo, "level": v, "n": int(w.sum()), "events": int(d.loc[w, y].sum()),
                            "pct": 100 * d.loc[w, y].mean()})
    pd.DataFrame(grp).to_csv(TAB / "X_grouped_crude.csv", index=False)

    # ---- 2. sequential confounder adjustment
    SETS = [("crude", []), ("+ age, sex", ["age", "female"]), ("+ hypermobility", ["age", "female", "hyper"]),
            ("+ disc displacement", ["age", "female", "hyper", "DISC"])]
    seq = []
    for y in OUTS:
        for expo in ["PAINFUL", "BURDEN"]:
            for name, cov in SETS:
                o, lo, hi, p = firth(d, y, [expo] + cov)[expo]
                seq.append({"outcome": y, "exposure": expo, "adjustment": name, "OR": o, "lo": lo, "hi": hi, "p": p})
    seq = pd.DataFrame(seq); seq.to_csv(TAB / "X_confounder_adjustment.csv", index=False)

    # ---- 3. subgroups with interaction tests (painful TMD vs none; fully adjusted for the other covariates)
    SUBS = [("female", "Sex", {0: "Men", 1: "Women"}), ("adult", "Age", {0: "<18 years", 1: ">=18 years"}),
            ("age40", "Age", {0: "<40 years", 1: ">=40 years"}), ("hyper", "Joint hypermobility", {0: "No", 1: "Yes"}),
            ("DISC", "Disc displacement", {0: "No", 1: "Yes"})]
    d["adult"] = (d.age >= 18).astype(float); d["age40"] = (d.age >= 40).astype(float)
    sub, inter = [], []
    for y in OUTS:
        for var, title, labs in SUBS:
            cov = [c for c in ["age", "female", "hyper", "DISC"] if c != var and not (var in ("adult", "age40") and c == "age")]
            for lev, lab in labs.items():
                s = d[d[var] == lev].reset_index(drop=True)
                o, lo, hi, p = firth(s, y, ["PAINFUL"] + cov)["PAINFUL"]
                sub.append({"outcome": y, "subgroup": title, "level": lab, "n": len(s),
                            "n_no_painful": int((s.PAINFUL == 0).sum()), "OR": o, "lo": lo, "hi": hi, "p": p})
            dd = d.assign(INT=d.PAINFUL * d[var])
            base = ["PAINFUL", var, "INT"] + [c for c in cov if c != var]
            inter.append({"outcome": y, "subgroup": f"{title} ({labs[0]} vs {labs[1]})", "p_interaction": firth(dd, y, base)["INT"][3]})
    inter = pd.DataFrame(inter)
    inter["p_interaction_holm"] = inter.groupby("outcome").p_interaction.transform(lambda p: multipletests(p, method="holm")[1])
    sub = pd.DataFrame(sub); sub.to_csv(TAB / "X_subgroups.csv", index=False); inter.to_csv(TAB / "X_interactions.csv", index=False)

    # ---- figure: subgroup forest (one per outcome, square, Okabe-Ito)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    for y, col in [("NP_clin", "#000000"), ("NP_move", "#56B4E9")]:
        s = sub[sub.outcome == y].reset_index(drop=True)
        overall = seq[(seq.outcome == y) & (seq.exposure == "PAINFUL") & (seq.adjustment == "+ disc displacement")].iloc[0]
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ys = np.arange(len(s) + 1)[::-1]
        ax.errorbar([overall.OR], [ys[0]], xerr=[[overall.OR - overall.lo], [overall.hi - overall.OR]], fmt="D", ms=4, color=col, capsize=2)
        ax.errorbar(s.OR, ys[1:], xerr=[s.OR - s.lo, s.hi - s.OR], fmt="o", ms=3, color=col, capsize=2)
        labels = ["All patients"] + [f"{r.subgroup}: {r.level.replace('>=', '≥')} (n={r.n})" for r in s.itertuples()]
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=6.5)
        ax.axvline(1, color="grey", lw=0.8, ls="--"); ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.3, 1, 3, 10, 30, 100]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Adjusted OR, painful TMD vs none (95% CI)")
        ax.set_title(f"Neck pain ({LAB[y].lower()}) by subgroup", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / f"F6_subgroups_painful_TMD_{y}.png"); plt.close(fig)

    # ---- 4. headache attributed to TMD, mutually adjusted with myalgia and joint pain
    hm = []
    for y in OUTS:
        for x, v in firth(d, y, ["MUSCLE", "JOINT", "HEAD", "age", "female", "hyper"]).items():
            hm.append({"outcome": y, "term": x, "OR": v[0], "lo": v[1], "hi": v[2], "p": v[3]})
    pd.DataFrame(hm).to_csv(TAB / "X_headache_model.csv", index=False)
    prof = pd.DataFrame({"M": d.MUSCLE, "J": d.JOINT, "H": d.HEAD})
    RES["headache_profiles"] = [{"myalgia": int(k[0]), "joint": int(k[1]), "headache": int(k[2]), "n": len(g),
                                 "clin_pct": 100 * d.NP_clin[g.index].mean(), "move_pct": 100 * d.NP_move[g.index].mean()}
                                for k, g in prof.groupby(["M", "J", "H"])]
    RES["interactions"] = inter.round(4).to_dict("records")
    (HERE / "outputs" / "results_subgroups.json").write_text(json.dumps(RES, indent=1), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps({k: v for k, v in RES.items() if k != "interactions"}))
    for t in ["X_grouped_crude", "X_confounder_adjustment", "X_subgroups", "X_interactions"]:
        print(f"\n== {t}"); print(pd.read_csv(TAB / f"{t}.csv").round(3).to_string(index=False))


def revision_figures():
    """Section from former make_figures_revision.py."""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    HERE = Path(__file__).parent
    T, FIG = HERE / "outputs" / "tables", HERE / "outputs" / "figures"
    OI = {"black": "#000000", "sky": "#56B4E9", "blue": "#0072B2", "orange": "#E69F00", "verm": "#D55E00", "grey": "#999999"}
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})


    def wilson(k, n, z=1.96):
        p = k / n; den = 1 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
        return p, c - h, c + h


    def logaxis(a):
        a.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 1.5, 2, 3, 4]))
        a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_formatter(matplotlib.ticker.NullFormatter())


    # ---- F1 burden gradient
    g = pd.read_csv(T / "X_grouped_crude.csv")
    g = g[g.exposure == "BURDEN"]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    for k, (y, col, lab) in enumerate([("NP_clin", OI["black"], "Movement or palpation (clinical)"),
                                       ("NP_move", OI["sky"], "Pain on active movement")]):
        s = g[g.outcome == y].sort_values("level")
        p, lo, hi = wilson(s.events.to_numpy(), s.n.to_numpy())
        x = s.level.to_numpy() + (-0.18 if k == 0 else 0.18)
        ax.bar(x, 100 * p, width=0.34, color=col, label=lab)
        ax.errorbar(x, 100 * p, yerr=[100 * (p - lo), 100 * (hi - p)], fmt="none", ecolor="grey", capsize=2, lw=0.8)
    n = g[g.outcome == "NP_clin"].sort_values("level").n.to_numpy()
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"None\n(n={n[0]})", f"1\n(n={n[1]})", f"2\n(n={n[2]})", f"3\n(n={n[3]})"])
    ax.set_xlabel("Painful TMD diagnoses (myalgia, joint pain, TMD headache)")
    ax.set_ylabel("Patients with neck pain (%)"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Neck pain by painful TMD burden", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "F1_neck_pain_by_painful_TMD_burden.png"); plt.close(fig)

    # ---- F2 specification curves with plain labels
    mv = pd.read_csv(T / "S_multiverse_64_specifications.csv")
    OUTL = {"NP_clin": ("Clinical (movement or palpation)", OI["black"]), "NP_palp": ("Palpation only", OI["orange"]),
            "NP_move": ("Movement only", OI["sky"]), "NP_strict": ("Movement and palpation", OI["blue"])}
    PRIMARY = ("NP_clin", "MUSCLE", "JOINT", "DISC", "base")
    for dx, name, col_ in [("MUSCLE", "Myalgia", "muscle_coding"), ("JOINT", "Joint pain", "joint_coding"),
                           ("DISC", "Disc displacement", "disc_coding")]:
        s = mv.sort_values(f"OR_{dx}").reset_index(drop=True)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
        for r, row in s.iterrows():
            prim = (row.outcome, row.muscle_coding, row.joint_coding, row.disc_coding, row.covariates) == PRIMARY
            a1.scatter(r, row[f"OR_{dx}"], s=24 if prim else 9, color=OUTL[row.outcome][1], marker="D" if prim else "o",
                       edgecolor="k" if prim else "none", zorder=3)
            if row[f"p_{dx}"] < 0.05:
                a1.scatter(r, row[f"OR_{dx}"], s=40, facecolor="none", edgecolor=OI["verm"], lw=0.7)
        a1.axhline(1, color="grey", lw=0.8, ls="--"); a1.set_yscale("log"); logaxis(a1.yaxis); a1.set_ylabel("Adjusted OR")
        a1.set_title(f"{name} (median OR {np.exp(np.log(s[f'OR_{dx}']).median()):.2f})", fontsize=8)
        ind = [(lab, (lambda o: lambda row: row.outcome == o)(o)) for o, (lab, _) in OUTL.items()] + \
              [("Broader diagnostic coding", lambda row: str(row[col_]).endswith("_alt")),
               ("Adjusted for hypermobility", lambda row: row.covariates == "hyper")]
        for k, (lab, f) in enumerate(ind):
            for r, row in s.iterrows():
                if f(row):
                    a2.scatter(r, k, s=5, marker="s", color=OUTL.get(row.outcome, (None, OI["black"]))[1] if k < 4 else OI["black"])
        a2.set_yticks(range(len(ind))); a2.set_yticklabels([i[0] for i in ind], fontsize=6.5); a2.invert_yaxis()
        a2.set_xticks([]); a2.set_xlabel("Specification (ranked by OR)")
        fig.tight_layout(); fig.savefig(FIG / f"F2_specification_curve_{dx.lower()}_64_definitions.png"); plt.close(fig)

    # ---- F7 DAG
    fig, ax = plt.subplots(figsize=(3.5, 3.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    nodes = {"TMD": (2.2, 5.0, "Painful TMD\n(myalgia, joint pain,\nTMD headache)"),
             "NP": (7.8, 5.0, "Neck pain"),
             "C": (5.0, 8.6, "Age, sex,\njoint hypermobility"),
             "U": (5.0, 1.4, "Unmeasured: central\nsensitization, distress,\nwidespread pain")}
    for k, (x, y, t) in nodes.items():
        dashed = k == "U"
        ax.add_patch(FancyBboxPatch((x - 1.9, y - 0.95), 3.8, 1.9, boxstyle="round,pad=0.1", fc="white",
                                    ec=OI["grey"] if dashed else OI["black"], ls="--" if dashed else "-", lw=1))
        ax.text(x, y, t, ha="center", va="center", fontsize=7, color=OI["grey"] if dashed else OI["black"])
    arrow = dict(arrowstyle="-|>", lw=1, color=OI["black"], shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(5.8, 5.35), xytext=(4.2, 5.35), arrowprops=arrow)
    ax.annotate("", xy=(4.2, 4.65), xytext=(5.8, 4.65), arrowprops=arrow)
    ax.text(5.0, 6.35, "trigeminocervical convergence", ha="center", va="bottom", fontsize=6)
    ax.text(5.0, 3.65, "cervical input, posture", ha="center", va="top", fontsize=6)
    for tgt in ("TMD", "NP"):
        x, y, _ = nodes[tgt]
        ax.annotate("", xy=(x + (0.6 if tgt == "TMD" else -0.6), y + 1.05), xytext=(5.0 + (-1.2 if tgt == "TMD" else 1.2), 7.6),
                    arrowprops=arrow)
        ax.annotate("", xy=(x + (0.6 if tgt == "TMD" else -0.6), y - 1.05), xytext=(5.0 + (-1.2 if tgt == "TMD" else 1.2), 2.4),
                    arrowprops=dict(arrow, color=OI["grey"], ls="--"))
    ax.set_title("Assumed causal relations (both directions possible;\ncross-sectional data cannot distinguish them)", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "F7_DAG_TMD_neck_pain.png"); plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    check_firth()
    prespecified_and_multiverse()
    revision_analyses()
    grouped_and_subgroups()
    revision_figures()
    print("all analyses done")
