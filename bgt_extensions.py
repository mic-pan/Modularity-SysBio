import BondGraphTools as bgt
from BondGraphTools.atomic import EqualFlow
from BondGraphTools import BondGraph
from BondGraphTools.reaction_builder import Reaction_Network
from BondGraphTools.algebra import adjacency_to_dict, inverse_coord_maps, get_relations_iterator
from BondGraphTools.algebra import _generate_substitutions, smith_normal_form

import sympy as sp
from sympy import SparseMatrix, lambdify, Symbol, simplify
import numpy as np
from matplotlib import pyplot as plt

def full_equations(model):
    "Returns full equations of a BondGraphTools model"
    X, mapping, A, F, G = model.system_model()
    # AX + F(X) = 0
    # G(X) = 0
    AX = A*SparseMatrix(X) + F
    full_model_equations = {}
    for i in range(AX.rows):
        xi = X[i]
        eqn = xi - AX[i,0]
        full_model_equations[str(xi)] = eqn
    return full_model_equations

def port_expressions(port,model):
    "Returns mathematical expressions corresponding to the effort and flow of bond at a port"
    mapping = model.system_model()[1]
    bond_index = mapping[1][port]
    full_model_equations = full_equations(model)
    return full_model_equations[f'e_{bond_index}'],full_model_equations[f'f_{bond_index}']

def get_ports(component):
    "Returns the ports of a component"
    return list(component.ports.keys())

def reaction_expressions(model,reaction):
    "Returns the mathematical expression corresponding to a reaction within a model"
    if type(reaction) == str:
        comp = model/f"R:{reaction}"
    elif isinstance(reaction,bgt.atomic.BondGraphBase):
        comp = reaction
    port1 = get_ports(comp)[0]
    port2 = get_ports(comp)[1]
    return port_expressions(port1,model),port_expressions(port2,model)

def reaction_flux_expression(model,reaction):
    return reaction_expressions(model,reaction)[0][1]

def reaction_affinity_expressions(model,reaction):
    expresions = reaction_expressions(model,reaction)
    return expresions[0][0],expresions[1][0]

def evaluate_bond_variable(model,expression,sol):
    states = [Symbol(x) for x in model.state_vars.keys()]
    f = lambdify(([states]),expression)
    if type(sol) == list:
        return np.array([f(x) for x in sol])
    else:
        return np.array([f(x) for x in sol.u])

def compute_fluxes(model,reaction,sol):
    "Compute the fluxes of a reaction from the solution of a simulation"
    v = reaction_flux_expression(model,reaction)
    return evaluate_bond_variable(model,v,sol)

def compute_affinities(model,reaction,sol):
    Af,Ar = reaction_affinity_expressions(model,reaction)
    return evaluate_bond_variable(model,Af,sol),evaluate_bond_variable(model,Ar,sol)

def compute_power(model,reaction,sol):
    v = reaction_flux_expression(model,reaction)
    Af,Ar = reaction_affinity_expressions(model,reaction)
    return evaluate_bond_variable(model,(Af-Ar)*v,sol)

def pop(model,component):
    """
    Removes a component from a model, and returns the component as well as the
    dangling port and side of the original bond.
    """
    for b in model.bonds:
        head_comp = b.head[0]
        tail_comp = b.tail[0]
        if (head_comp is component):
            side = "tail"
            if isinstance(tail_comp,bgt.atomic.PortExpander):
                port = tail_comp
            else:
                port = b.tail
            bgt.disconnect(head_comp,tail_comp)
            break
        elif (tail_comp is component):
            side = "head"
            if isinstance(head_comp,bgt.atomic.PortExpander):
                port = head_comp
            else:
                port = b.head
            bgt.disconnect(head_comp,tail_comp)
            break

    bgt.remove(model,component)
    return component,(port,side)

def swap(old_component,new_component):
    """
    Swaps one component for another component. The new component cannot exist
    within the same model
    """
    model = old_component.parent
    port,side = pop(model,old_component)[1]
    bgt.add(model,new_component)
    if side == "head":
        bgt.connect(new_component,port)
    elif side == "tail":
        bgt.connect(port,new_component)

    if isinstance(port,EqualFlow):
        check_flow_junction(port)
        
        
def expose(component,label=None):
    """
    Exposes a single-port component to make it available for external connection.
    This function is a quick fix for the corresponding function in BondGraphTools.
    """
    model = component.parent
    if label is None:
        label = str(len(model.ports))
        
    if component.__component__ == "SS":
        bgt.expose(component,label)
    else:
        SS = bgt.new("SS",name=component.name)
        swap(component,SS)
        bgt.expose(SS,label)

def check_flow_junction(component):
    model = component.parent
    bonds = model.bonds
    for b in bonds:
        if b.tail[0] == component:
            check_flow_port(b.tail,"tail")
        if b.head[0] == component:
            check_flow_port(b.head,"head")

def check_flow_port(port,side):
    if side == "tail":
        port.weight = 1
    elif side == "head":
        port.weight = -1

def map_state_to_component(component,variable):
    if isinstance(component,bgt.atomic.Component):
        return component
    else:
        inner_comp,inner_var = component.state_vars[variable]
        return map_state_to_component(inner_comp,inner_var)

def state_index(model,component):
    for i,(state,variable) in enumerate(model.state_vars.values()):
        if component is map_state_to_component(state,variable):
            return i
    print("Component not found in state variables.")
    return None

def extract_solution_values(sol,model,component,t=None):
    index = state_index(model,component)
    if t == None:
        return sol.t, np.array(sol.u)[:,index]
    else:
        return t, np.array(sol(t))[index,:]
    
def plot_solution(sol,model,component,t=None):
    t,x = extract_solution_values(sol,model,component,t)
    fig = plt.plot(t,x,label=component.uri)
    plt.xlabel("Time")
    plt.ylabel("Concentration")

class StaticBondGraph(BondGraph):
    """
    An alternative class for bond graph models that will only re-calculate constitutive relations
    when specified by the user.
    """
    def update_constitutive_relations(self):
        self.stored_constitutive_relations = super().constitutive_relations
    
    @property
    def constitutive_relations(self):
        try:
            result = self.stored_constitutive_relations
        except:
            self.update_constitutive_relations()
            result = self.stored_constitutive_relations
        return result

def generate_rn(reactions,name,chemostats=[]):
    """
    Makes a reaction network from a set of reactions and chemostats.
        reactions: a tuple of (reaction_string, reaction_name)
        name: name of the reaction network
        chemostats: contains the chemostats of the network
    """
    rn = Reaction_Network(name=name)
    for reaction,name in reactions:
        rn.add_reaction(reaction,name=name)
    for s in chemostats:
        rn.add_chemostat(s)
    return rn

def generate_enzyme_model(reactions,name):
    """
    Generates a bond graph of a chemical reaction network.
        reactions: a tuple of (reaction_string, reaction_name)
        name: name of the reaction network
    """
    rn = generate_rn(reactions,name)
    model = rn.as_network_model(normalised=True)
    return model

def add_reactions(system,reactions,chemostats=[],normalised=False):
    """
    Adds reactions given the the reaction network to the bond graph model given by system.
        system: the bond graph model to which the components are added to
        reactions: a tuple of (reaction_string, reaction_name)
        name: name of the reaction network
        chemostats: contains the chemostats of the network
    """
    rn = generate_rn(reactions,name=system.name,chemostats=chemostats)
    species_anchor = rn._build_species(system, normalised)
    rn._build_reactions(system, species_anchor, normalised)

def find(model,uri):
    """
    Finds the component in a model given by a (relative) URI
    Example: find(model,"Module/Inner module/"C:component")
    """
    index = uri.find("/")
    if index < 0:
        return model/uri
    else:
        comp_str = uri[:index]
        remaining_path = uri[index+1:]
        return find(model/comp_str, remaining_path)
    
def set_parameters(model,parameters):
    """
    Set parameters for a bond graph model.
        model: the bond graph model to which parameters are set
        parameters: a dict (component_string,parameter_string) -> parameter_value
    """
    for (comp,p),v in parameters.items():
        try:
            component = find(model,comp)
            if (component.template) != "base/SS":
                component.set_param(p,v)
        except:
            pass

def set_component_parameters(parameter_dict):
    """
    Sets the parameters of the components specified in the keys of the argument to the values.
    parameter_dict: (component, parameter_string) -> parameter_value
    """
    for (comp,p),v in parameter_dict:
        try:
            if (comp.template) != "base/SS":
                (model/comp).set_param(p,v)
        except:
            pass

def Ce(**kwargs):
    """
    Creates a bond graph Ce component
    """
    return bgt.new(library="BioChem",component="Ce",**kwargs)

def Re(**kwargs):
    """
    Creates a bond graph Re component
    """
    return bgt.new(library="BioChem",component="Re",**kwargs)

def Se(**kwargs):
    """
    Creates a bond graph Se component
    """
    return bgt.new("Se",**kwargs)

def unify(species,unified_name):
    """
    Merges species components into a single component. The species must all exist within modules at the same level,
    for example:
        model/submodule1/C:species1
        model/submodule2/C:species2
    
    Arguments:
        species: a tuple containing components correponding to the equivalent species
        unified_name: a string for the name of the unified component
    """
    # Check that the species all exist on the same level
    models = [s.parent for s in species]
    if all([m.parent is models[0].parent for m in models]):
        parent_model = models[0].parent
    else:
        raise ValueError("The species must all be on the same level of the model")
    
    # Check that the species have the same model type
    if all([s.template == species[0].template for s in species]):
        model_type = species[0].template
    else:
        raise ValueError("The species must have the same model type")
        
    # Expose the species
    port_names = []
    for s in species:
        port_names.append(s.name)
        expose(s,s.name)
        
    # Add the unified component
    if model_type == "BioChem/Ce":
        merged_species = Ce(name=unified_name)
    elif model_type == "base/Se":
        merged_species = Se(name=unified_name)
        
    if len(species) == 1:
        bgt.add(parent_model,merged_species)
        anchor = merged_species
    else:
        junction = bgt.new("0",name=unified_name)
        bgt.add(parent_model,merged_species,junction)
        bgt.connect(junction,merged_species)
        anchor = junction
    
    # Connect the components together
    for m,port in zip(models,port_names):
        bgt.connect((m,port),anchor)
        
    return merged_species,anchor

def promote(component,n=1):
    """
    Moves a component (with a single connection) a level higher in the hierarchy
    """
    if n == 0:
        return
    else:
        parent = component.parent
        parent_of_parent = parent.parent
        expose(component,component.name)
        bgt.add(parent_of_parent,component)
        bgt.connect((parent,component.name),component)
        promote(component,n-1)
        
def set_aqueous_solution_parameters(model,R=8.314,T=310):
    """
    Sets the gas constant (R) and temperature (T) for a bond graph model.
    """
    for comp in model.components:
        if (comp.metamodel == "C") and (comp.template == "BioChem/Ce"):
            comp.set_param("R",R)
            comp.set_param("T",T)

def reduce_model(linear_op, nonlinear_op, coordinates, size_tuple,
                 control_vars=None):
    """
    An alternative method for the reducing a model in BondGraphTools. It 
    first uses the Smith normal form to reduce the linear constrants and 
    then attempts use substitutions to define the remaining nonlinear
    constraints.
    """
    reduced_linear_op, reduced_nonlinear_op, constraints = smith_normal_form(linear_op,nonlinear_op)
    nonlinear_arguments = reduced_nonlinear_op.atoms()
    bond_vars = [x for x in list(nonlinear_arguments) if isinstance(x,Symbol) and x.name[0] in ['e','f']]
    
    for x in bond_vars:
        idx = coordinates.index(x)
        bond_expr = reduced_linear_op[idx,:]*SparseMatrix(coordinates)+reduced_nonlinear_op[idx,:]
        x_expression = sp.solve(bond_expr[0],x)[0]
        for i,expr in enumerate(reduced_nonlinear_op):
            reduced_nonlinear_op[i] = expr.subs(x,x_expression)

    reduced_nonlinear_op = simplify(reduced_nonlinear_op)
    
    return coordinates, reduced_linear_op, reduced_nonlinear_op, constraints

def reduce_model2(linear_op, nonlinear_op, coordinates, size_tuple, external_port_indices,
                  control_vars=None):
    variables_dict = {}
    equations = linear_op*SparseMatrix(coordinates)+nonlinear_op
    #TODO: Fix this so that the port vars only correspond to the component
    port_vars = set([Symbol("e_"+str(i)) for i in external_port_indices])

    flag = True
    max_iters = 200
    i = 0
    while flag and i<max_iters:
        equations,flag = update(equations,variables_dict,port_vars)
        i += 1

    n = len(coordinates)
    reduced_linear_op = SparseMatrix(sp.zeros(n,n))
    reduced_nonlinear_op = SparseMatrix(sp.zeros(n,1))
    for i,z in enumerate(coordinates):
        if z in variables_dict.keys():
            reduced_linear_op[i,i] = 1
            reduced_nonlinear_op[i] = -variables_dict[z]

    constraints = []
    return coordinates, reduced_linear_op, reduced_nonlinear_op, variables_dict, constraints

class BondGraphBioChem(BondGraph):
    """
    A BondGraph class for biochemistry that attempts to fix issues that arise
    from complex rate laws.
    """
    def _system_model(self, control_vars=None):
        mappings, coordinates = inverse_coord_maps(
            *self._build_internal_basis_vectors()
        )
        inv_tm, inv_js, inv_cv = mappings

        js_size = len(inv_js)  # number of ports
        ss_size = len(inv_tm)  # number of state space coords
        cv_size = len(inv_cv)
        n = len(coordinates)

        size_tuple = (ss_size, js_size, cv_size, n)

        lin_dict = adjacency_to_dict(inv_js, self.bonds, offset=ss_size)

        nlin_dict = {}

        try:
            row = max(row + 1 for row, _ in lin_dict.keys())
        except ValueError:
            row = 0

        inverse_port_map = {}

        for port, (cv_e, cv_f) in self._port_map.items():
            inverse_port_map[cv_e] = ss_size + 2*inv_js[port]
            inverse_port_map[cv_f] = ss_size + 2*inv_js[port] + 1
        
        for component in self.components:
            relations = get_relations_iterator(
                component, mappings, coordinates, inverse_port_map
            )

            for linear, nonlinear in relations:
                lin_dict.update({(row, k): v
                                 for k, v in linear.items()})
                nlin_dict.update({(row, 0): nonlinear})
                row += 1

        linear_op = SparseMatrix(row, n, lin_dict)
        nonlinear_op = SparseMatrix(row, 1, nlin_dict)

        external_port_indices = [i for p,i in inv_js.items() if p not in self.internal_ports]
        
        coordinates, linear_op, nonlinear_op, variables_dict, constraints = reduce_model2(
                linear_op, nonlinear_op, coordinates, size_tuple, external_port_indices,
            control_vars=control_vars
        )
        return coordinates, mappings, linear_op, nonlinear_op, variables_dict, constraints

    def system_model(self, control_vars=None):
        coordinates, mappings, linear_op, nonlinear_op, variables_dict, constraints = self._system_model(
            control_vars
        )
        return coordinates, mappings, linear_op, nonlinear_op, constraints

    @property
    def constitutive_relations(self):
        coordinates, mappings, lin_op, nlin_op, variables_dict, constraints = self._system_model()
        inv_tm, inv_js, _ = mappings
        out_ports = [idx for p, idx in inv_js.items() if p in self.ports]
        js_size = len(inv_js)  # number of ports
        ss_size = len(inv_tm)  # number of state space coords
        
        # Extract only the constraints relating to rates
        #equations = lin_op*SparseMatrix(coordinates)+nlin_op
        rates = [Symbol("dx_"+str(i)) for i in range(len(self.state_vars))]
        rate_equations = [dx-variables_dict[dx] for dx in rates]

        port_vars = [Symbol("f_"+str(i)) for i in out_ports]
        port_equations = [f-variables_dict[f] for f in port_vars]

        equations = rate_equations + port_equations

        subs = []
        for local_idx, c_idx in enumerate(out_ports):
            p, = {pp for pp in self.ports if pp.index == local_idx}
            label = p.index
            subs.append(sp.symbols((f"e_{c_idx}", f"e_{label}")))
            subs.append(sp.symbols((f"f_{c_idx}", f"f_{label}")))

        return [r.subs(subs) for r in equations]

class StaticBGBioChem(BondGraphBioChem):
    def update_constitutive_relations(self):
        self.stored_constitutive_relations = super().constitutive_relations
    
    @property
    def constitutive_relations(self):
        try:
            result = self.stored_constitutive_relations
        except:
            self.update_constitutive_relations()
            result = self.stored_constitutive_relations
        return result

def _computed_vars(expr,port_vars=None):
    nonlinear_arguments = expr.atoms()
    v = [x for x in list(nonlinear_arguments-port_vars) if isinstance(x,Symbol) and x.name[0] in ['e','f','d']]
    return v

def update(equations,variables_dict,port_vars):
    flag = False
    for i,x in enumerate(equations):
        v = _computed_vars(x,port_vars)
        if len(v) == 1:
            var = v[0]
            sol = quick_solve(x,var)
            variables_dict[var] = sol
            equations[i] = 0
            equations = equations.subs(var,sol)
            #equations = [sp.factor(y.subs(var,sol)) for y in equations]
            flag = True
    return equations,flag

def quick_solve(expr,x):
    """
    This function will solve a symbolic expression for a variable, assuming that the variable
    only appears once as a linear term in the expression.
    """
    c = expr.coeff(x)
    scaled_answer = expr-c*x
    assert x not in scaled_answer.atoms()
    return -scaled_answer/c

def gather_cv(model):
    """
    Gathers the control variables of a bond graph model with a simulation_cvs attribute.
    The simulation_cvs should follow the following structure:
        (component, parameter string): parameter value
    """
    control_vars = {}
    for k,v in recursive_cvs(model).items():
        control_vars[k] = model.simulation_cvs[v]
    return control_vars

def recursive_cvs(model):
    cv = model.control_vars
    result = {}
    for k,(c,p) in cv.items():
        comp = c
        var = p
        while not isinstance(comp,bgt.atomic.Component):
            entry = comp.control_vars[var]
            comp = entry[0]
            var = entry[1]
        result[k] = (comp,var)
    return result