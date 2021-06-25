import BondGraphTools as bgt
from BondGraphTools.exceptions import ModelException

from julia import Main
from diffeqpy import de

def simulate(system,tspan,x0,dx0=None,control_vars=None,**kwargs):
    """
    A new simulate function for BondGraphTools models that uses the most recent
    versions of the julia and diffeqpy packages. This function also allows options
    to the solvers to be specified.
    """
    if system.ports:
        raise ModelException(
            "Cannot Simulate %s: unconnected ports %s",
            system, system.ports)

    if system.control_vars and not control_vars:
        raise ModelException("Control variable not specified")
    
    julia_func_str,diffs = bgt.sim_tools.to_julia_function_string(system,control_vars)
    func = Main.eval(julia_func_str)
    X0, DX0 = bgt.sim_tools._fetch_ic(x0, dx0, system, func)
    problem = de.DAEProblem(func, DX0, X0, tspan, differential_vars=diffs,**kwargs)
    sol = de.solve(problem)
    return sol