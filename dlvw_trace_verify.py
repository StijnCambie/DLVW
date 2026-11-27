#!/usr/bin/env python3
"""Generate and verify the finite DLVW trace table.

This is the single computational entry point for the accompanying manuscript.
It performs two independent tasks:

1. It enumerates the reduced two-geodesic traces described in the paper.
   There are 8 upper-stem traces, 9 direct/direct traces, and 81 side/side
   traces, hence 98 labelled traces.  Reflection interchanges the two diameter
   endpoints and leaves 55 orbits.  The script writes both a CSV file and two
   LaTeX longtables (all labelled traces and one representative per orbit).

2. It verifies the exact rational Farkas certificates for the R12 master
   system.  The certificate archive was generated with a numerical LP solver,
   but verification below uses only Python's standard library and exact
   ``fractions.Fraction`` arithmetic.  Thus no floating-point result is trusted.

Usage:
    python dlvw_trace_verify.py

All output files are written next to this script.  The script exits with a
nonzero status as soon as a count, orbit partition, or certificate identity
fails.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
CERT_PATH = HERE / "dlvw_r12_certificates.json"
FULL_TEX = HERE / "trace_table_full.tex"
ORBIT_TEX = HERE / "trace_table_orbits.tex"
CSV_PATH = HERE / "trace_table.csv"
REPORT_PATH = HERE / "verification_report.txt"


# ---------------------------------------------------------------------------
# Part I.  The trace census
# ---------------------------------------------------------------------------

GOOD_GATES = (
    ("G0", r"arm interior"),
    ("G1", r"arm--diameter junction"),
    ("G2", r"diameter tail"),
)

WRONG_INTERVALS = (
    ("W0", r"single wrong-side contact"),
    ("W1", r"wrong-side interval ending toward the branch vertex"),
    ("W2", r"wrong-side interval ending toward the diameter junction"),
)

UPPER_COMPANIONS = (
    ("C0", r"branch vertex"),
    ("C1", r"good-arm interior"),
    ("C2", r"arm--diameter junction"),
    ("C3", r"diameter tail"),
)


@dataclass(frozen=True, order=True)
class Trace:
    """One labelled pair of normalised shortest-path traces."""

    family: str
    left: str
    right: str

    @property
    def trace_id(self) -> str:
        if self.family == "U":
            return f"{self.left}-{self.right}"
        return f"{self.family}-{self.left}-{self.right}"

    @property
    def reduction(self) -> str:
        return {
            "U": "upper-stem template",
            "GG": "direct/direct template",
            "BB": "side/side template",
        }[self.family]

    @property
    def forced_tail(self) -> str:
        if self.family == "U":
            return "the companion path follows its good geodesic"
        if self.family == "GG":
            return "both paths follow their good geodesics"
        return "after the second gate, both paths follow their good geodesics"

    def reflect(self) -> "Trace":
        """Interchange v_0 and v_d.

        The local mode alphabets were deliberately chosen symmetrically, so
        reflection simply swaps the two endpoint traces.  For the upper-stem
        family it swaps which endpoint path uses the upper stem.
        """
        if self.family == "U":
            endpoint = "Ud" if self.left == "U0" else "U0"
            return Trace("U", endpoint, self.right)
        return Trace(self.family, self.right, self.left)


def enumerate_traces() -> list[Trace]:
    traces: list[Trace] = []

    # One endpoint path reaches the upper stem.  The other path is direct and
    # has at most four possible first-contact gates.  There are two endpoint
    # choices, hence 2*4 = 8 labelled traces.
    for endpoint in ("U0", "Ud"):
        for companion, _ in UPPER_COMPANIONS:
            traces.append(Trace("U", endpoint, companion))

    # Both endpoint paths go directly in their good directions: 3*3 = 9.
    for left, _ in GOOD_GATES:
        for right, _ in GOOD_GATES:
            traces.append(Trace("GG", left, right))

    # A side path may first use one wrong-side interval (three modes) and then
    # enter the good side through one of three gates.  Thus each endpoint has
    # 3*3 = 9 modes and an ordered pair has 9^2 = 81 traces.
    bad_modes = [f"{wrong}{gate}" for wrong, _ in WRONG_INTERVALS for gate, _ in GOOD_GATES]
    for left in bad_modes:
        for right in bad_modes:
            traces.append(Trace("BB", left, right))

    order = {"U": 0, "GG": 1, "BB": 2}
    traces.sort(key=lambda t: (order[t.family], t.left, t.right))
    return traces


def orbit_representatives(traces: Iterable[Trace]) -> list[tuple[Trace, int]]:
    remaining = set(traces)
    reps: list[tuple[Trace, int]] = []
    while remaining:
        t = min(remaining)
        orbit = {t, t.reflect()}
        present = orbit & remaining
        remaining -= present
        reps.append((min(orbit), len(orbit)))
    order = {"U": 0, "GG": 1, "BB": 2}
    reps.sort(key=lambda pair: (order[pair[0].family], pair[0].left, pair[0].right))
    return reps


def describe_endpoint(code: str) -> str:
    if code in {name for name, _ in GOOD_GATES}:
        return dict(GOOD_GATES)[code]
    if code in {name for name, _ in UPPER_COMPANIONS}:
        return dict(UPPER_COMPANIONS)[code]
    if len(code) == 4 and code[:2] in dict(WRONG_INTERVALS) and code[2:] in dict(GOOD_GATES):
        return f"{dict(WRONG_INTERVALS)[code[:2]]}; then {dict(GOOD_GATES)[code[2:]]}"
    if code == "U0":
        return r"the $v_0$-path reaches the upper stem"
    if code == "Ud":
        return r"the $v_d$-path reaches the upper stem"
    raise KeyError(code)


def tex_escape_plain(text: str) -> str:
    return text.replace("--", r"--")


def write_full_table(traces: list[Trace]) -> None:
    lines = [
        r"\begin{longtable}{@{}r >{\raggedright\arraybackslash}p{.13\textwidth} >{\raggedright\arraybackslash}p{.22\textwidth} >{\raggedright\arraybackslash}p{.22\textwidth} >{\raggedright\arraybackslash}p{.23\textwidth}@{}}",
        r"\caption{The 98 labelled two-geodesic traces.  The notation is defined in Section~\ref{sec:trace-alphabet}.}\label{tab:full-traces}\\",
        r"\toprule",
        r"No. & ID & $Q_0$ trace & $Q_d$ trace & Reduction block\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"No. & ID & $Q_0$ trace & $Q_d$ trace & Reduction block\\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{5}{r}{Continued on the next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for i, t in enumerate(traces, start=1):
        if t.family == "U":
            if t.left == "U0":
                q0 = describe_endpoint("U0")
                qd = describe_endpoint(t.right)
            else:
                q0 = describe_endpoint(t.right)
                qd = describe_endpoint("Ud")
        else:
            q0 = describe_endpoint(t.left)
            qd = describe_endpoint(t.right)
        lines.append(rf"{i} & \texttt{{{t.trace_id}}} & {q0} & {qd} & {t.reduction}\\")
    lines.append(r"\end{longtable}")
    FULL_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_orbit_table(reps: list[tuple[Trace, int]]) -> None:
    lines = [
        r"\begin{longtable}{@{}r >{\raggedright\arraybackslash}p{.22\textwidth} >{\raggedright\arraybackslash}p{.20\textwidth} >{\raggedright\arraybackslash}p{.20\textwidth} c@{}}",
        r"\caption{One representative of each reflection orbit.}\label{tab:trace-orbits}\\",
        r"\toprule",
        r"No. & Representative & First endpoint mode & Second endpoint mode & Orbit size\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"No. & Representative & First endpoint mode & Second endpoint mode & Orbit size\\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{5}{r}{Continued on the next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for i, (t, size) in enumerate(reps, start=1):
        lines.append(
            rf"{i} & \texttt{{{t.trace_id}}} & \texttt{{{t.left}}} & "
            rf"\texttt{{{t.right}}} & {size}\\"
        )
    lines.append(r"\end{longtable}")
    ORBIT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(traces: list[Trace]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["number", "id", "family", "v0_trace", "vd_trace", "reduction", "reflection"])
        for i, t in enumerate(traces, start=1):
            writer.writerow([i, t.trace_id, t.family, t.left, t.right, t.reduction, t.reflect().trace_id])


# ---------------------------------------------------------------------------
# Part II.  Exact verification of the R12 certificate archive
# ---------------------------------------------------------------------------


def F(value) -> Fraction:
    return Fraction(str(value))


def add(a: dict[str, Fraction], b: dict[str, Fraction]) -> dict[str, Fraction]:
    ans = dict(a)
    for key, value in b.items():
        ans[key] = ans.get(key, Fraction(0)) + value
        if ans[key] == 0:
            del ans[key]
    return ans


def scale(c: Fraction, a: dict[str, Fraction]) -> dict[str, Fraction]:
    return {key: c * value for key, value in a.items() if c * value}


def vec(**kw: int) -> dict[str, Fraction]:
    return {key: Fraction(value) for key, value in kw.items() if value}


def parse_vec(obj: dict[str, str]) -> dict[str, Fraction]:
    return {key: F(value) for key, value in obj.items() if F(value)}


def branch_system(branch, base_ge, base_eq):
    b15, b16, bf1, bf2 = branch
    ges = dict(base_ge)
    eqs = dict(base_eq)

    A = vec(x11=1, x10=1, x7=1, x3=1)
    B = vec(x1=1, x4=1)
    M, O = (A, B) if b15 == 0 else (B, A)
    ges["M15choice"] = (add(M, scale(Fraction(-1), O)), Fraction(0))
    eqs["D15"] = (add(vec(x15=1, d=-1, L=1), M), Fraction(0))

    C = vec(x11=1, x10=1, x1=1, x4=1)
    D = vec(x7=1, x3=1)
    M2, O2 = (C, D) if b16 == 0 else (D, C)
    ges["M16choice"] = (add(M2, scale(Fraction(-1), O2)), Fraction(0))
    eqs["D16"] = (add(vec(x16=1, d=-1, L=1), M2), Fraction(0))

    E, Ec = vec(x6=1, x16=-1), Fraction(-1)
    FF, Fc = vec(x8=1), Fraction(0)
    Sel, Sc, Other, Oc = (E, Ec, FF, Fc) if bf1 == 0 else (FF, Fc, E, Ec)
    ges["F1maxchoice"] = (add(Sel, scale(Fraction(-1), Other)), Oc - Sc)
    ges["F1fail"] = (add(Sel, vec(x5=1, x15=-1)), Fraction(1) - Sc)

    G, Gc = vec(x5=1, x15=-1), Fraction(-1)
    H, Hc = vec(x8=1), Fraction(0)
    Sel, Sc, Other, Oc = (G, Gc, H, Hc) if bf2 == 0 else (H, Hc, G, Gc)
    ges["F2maxchoice"] = (add(Sel, scale(Fraction(-1), Other)), Oc - Sc)
    ges["F2fail"] = (add(Sel, vec(x6=1, x16=-1)), Fraction(1) - Sc)

    for name in [f"x{i}" for i in range(1, 12)] + ["d", "L"]:
        ges[f"{name}>=0"] = (vec(**{name: 1}), Fraction(0))
    ges["x8>=1"] = (vec(x8=1), Fraction(1))
    return ges, eqs


def check_certificate(cert, ges, eqs) -> None:
    coef: dict[str, Fraction] = {}
    rhs = Fraction(0)
    for label, value in cert["inequality_multipliers"].items():
        lam = F(value)
        if lam < 0:
            raise AssertionError((label, lam))
        a, b = ges[label]
        coef = add(coef, scale(lam, a))
        rhs += lam * b
    for label, value in cert["equality_multipliers"].items():
        mu = F(value)
        a, b = eqs[label]
        coef = add(coef, scale(mu, a))
        rhs += mu * b
    target = parse_vec(cert["target"])
    target_rhs = F(cert["target_rhs"])
    if coef != target:
        raise AssertionError((coef, target))
    if rhs < target_rhs or rhs != F(cert["derived_rhs"]):
        raise AssertionError((rhs, target_rhs, cert["derived_rhs"]))


def verify_r12() -> tuple[int, int]:
    payload = json.loads(CERT_PATH.read_text(encoding="utf-8"))
    if payload["format"] != "DLVW-R12-Farkas-v1":
        raise AssertionError("unknown R12 certificate format")
    base_ge = {
        row["label"]: (parse_vec(row["coefficients"]), F(row["rhs"]))
        for row in payload["base_inequalities"]
    }
    base_eq = {
        row["label"]: (parse_vec(row["coefficients"]), F(row["rhs"]))
        for row in payload["base_equalities"]
    }
    exceptional = tuple(payload["exceptional_branch"])

    infeasible_count = 0
    certified = set()
    for key, cert in payload["infeasible_branches"].items():
        branch = tuple(int(x) for x in key.split(","))
        if branch == exceptional:
            raise AssertionError("exceptional branch incorrectly marked infeasible")
        ges, eqs = branch_system(branch, base_ge, base_eq)
        check_certificate(cert, ges, eqs)
        certified.add(branch)
        infeasible_count += 1

    all_branches = set(itertools.product([0, 1], repeat=4))
    if certified | {exceptional} != all_branches:
        raise AssertionError("the sixteen max branches are not completely covered")

    ges, eqs = branch_system(exceptional, base_ge, base_eq)
    equality_count = 0
    for _, pair in payload["exceptional_equalities"].items():
        check_certificate(pair["forward"], ges, eqs)
        check_certificate(pair["reverse"], ges, eqs)
        equality_count += 1

    if infeasible_count != 15 or equality_count != 8:
        raise AssertionError((infeasible_count, equality_count))
    return infeasible_count, equality_count


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    traces = enumerate_traces()
    if len(traces) != 98 or len(set(traces)) != 98:
        raise AssertionError("the labelled trace census must have size 98")

    family_counts = {family: sum(t.family == family for t in traces) for family in ("U", "GG", "BB")}
    if family_counts != {"U": 8, "GG": 9, "BB": 81}:
        raise AssertionError(family_counts)

    reps = orbit_representatives(traces)
    if len(reps) != 55 or sum(size for _, size in reps) != 98:
        raise AssertionError("reflection must give 55 orbits covering 98 traces")

    write_full_table(traces)
    write_orbit_table(reps)
    write_csv(traces)

    infeasible_count, equality_count = verify_r12()

    report = [
        "DLVW trace and R12 verification report",
        "======================================",
        "",
        f"labelled traces: {len(traces)}",
        f"  upper-stem: {family_counts['U']}",
        f"  direct/direct: {family_counts['GG']}",
        f"  side/side: {family_counts['BB']}",
        f"reflection orbits: {len(reps)}",
        "",
        f"R12 infeasible max branches verified exactly: {infeasible_count}",
        f"R12 forced exceptional equalities verified exactly: {equality_count}",
        "",
        f"trace_table.csv sha256: {sha256(CSV_PATH)}",
        f"trace_table_full.tex sha256: {sha256(FULL_TEX)}",
        f"trace_table_orbits.tex sha256: {sha256(ORBIT_TEX)}",
        f"certificate archive sha256: {sha256(CERT_PATH)}",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
