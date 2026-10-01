"""Adversarial benchmark corpus for the three parameter-geometry cost centers."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import sympy as sp
from semialg import parametric_cad

from odeanalysis import (
    FirstOrderSystem,
    leading_rank_analysis,
    local_parameter_analysis,
    newton_loci,
    singularity_loci,
    stokes_formal_loci,
    stokes_ray_loci,
)


@dataclass(frozen=True)
class Case:
    name: str
    parameters: tuple[sp.Symbol, ...]
    extract: Callable[[], tuple[sp.Expr, ...]]
    analyze: Callable[[], object]

def timed(call):
    start = perf_counter()
    value = call()
    return perf_counter() - start, value

def marker(loci):
    return sp.Ne(sp.prod(loci), 0, evaluate=False) if loci else sp.true

def corpus():
    x=sp.symbols('x')
    y=sp.Function('y')
    a,b,c,d,e=sp.symbols('a b c d e', real=True)
    cases=[]
    ode1=(a*x**5+b*x**4+c*x**3+x+1)*sp.diff(y(x),x,2)+y(x)
    cases.append(Case('nested_degree_loss',(a,b,c),lambda:singularity_loci(ode1,y,x),lambda:local_parameter_analysis(ode1,y,x,0)))
    ode2=x**7*sp.diff(y(x),x,3)+(a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+x)*sp.diff(y(x),x)+y(x)
    cases.append(Case('irrelevant_support_coefficients',(a,b,c,d,e),lambda:newton_loci(ode2,y,x),lambda:local_parameter_analysis(ode2,y,x,0)))
    ode3=x**4*sp.diff(y(x),x,2)+((a-b)*x**2+(a**2-b**2)*x)*sp.diff(y(x),x)+y(x)
    cases.append(Case('normalized_support_cancellation',(a,b),lambda:newton_loci(ode3,y,x),lambda:local_parameter_analysis(ode3,y,x,0)))
    system=FirstOrderSystem(x,sp.ImmutableMatrix([[1/x,0,0,0],[0,a/x,0,0],[0,0,b/x,0],[0,0,0,0]]))
    cases.append(Case('rank_det_identically_zero',(a,b),lambda:(a,b),lambda:leading_rank_analysis(system,0,(a,b))))
    ode5=(a*x**3+b*x**2+b*x+1)*sp.diff(y(x),x,2)+y(x)
    cases.append(Case('discriminant_meets_degree_loss',(a,b),lambda:singularity_loci(ode5,y,x),lambda:local_parameter_analysis(ode5,y,x,0)))
    ode6=x**2*sp.diff(y(x),x,2)+(a*b*c*d)*x*sp.diff(y(x),x)+(a*b*c*d)*y(x)
    cases.append(Case('many_cells_few_ode_invariants',(a,b,c,d),lambda:newton_loci(ode6,y,x),lambda:local_parameter_analysis(ode6,y,x,0)))
    t=sp.symbols('t')
    parts=(t**2,(a+sp.I*b)*t**2,(c+sp.I*d)*t**2)
    cases.append(Case('stokes_phase_without_formal_loss',(a,b,c,d),lambda:stokes_ray_loci(parts,t,(a,b,c,d)),lambda:stokes_formal_loci(parts,t)))
    ode8=x**3*sp.diff(y(x),x,2)+(a**3)*(x**2)*sp.diff(y(x),x)+(a**3*b**2)*y(x)
    cases.append(Case('redundant_reparameterized_loci',(a,b),lambda:newton_loci(ode8,y,x),lambda:local_parameter_analysis(ode8,y,x,0)))
    return tuple(cases)

def main():
    import sys
    selected = set(sys.argv[1:])
    print('case,transition_extraction_s,semialg_decomposition_s,ode_analysis_s,loci,cells', flush=True)
    for case in corpus():
        if selected and case.name not in selected:
            continue
        te,loci=timed(case.extract)
        ts,geometry=timed(
            lambda loci=loci, case=case: parametric_cad(
                marker(loci), (), parameters=case.parameters, output='result'
            )
        )
        ta,_=timed(case.analyze)
        print(f'{case.name},{te:.6f},{ts:.6f},{ta:.6f},{len(loci)},{len(geometry.cases)}', flush=True)
if __name__ == '__main__':
    main()
