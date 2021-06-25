import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

import BondGraphTools as bgt
from BondGraphTools import BondGraph
bgt.component_manager.load_library('enzymes.json')

import sys
import bgt_extensions as bgex
from bgt_extensions import Ce, Se, Re
from bgt_extensions import BondGraphBioChem, StaticBGBioChem
from bgt_juliasim import simulate

# Load the parameters from the Mason and Covert study
parameters = np.load("data/mason_covert_upper_sat_limits_1e2_pars.npy")
# Pick the set of parameters with shortest recovery time
p = parameters[:,265] 

RT_kcal = 0.6 # RT in kcal/mol (Note that this corresponds to T=301.5K rather than 310K as stated in Mason and Covert.)
R = 8.3144598 # Ideal gas constant in J/K/mol

def G_to_K(G0,RT=R*310):
    "Inverse log transformation of parameters"
    return np.exp(G0/RT)

# Define the default initial conditions for simulating the models
concentration_potentials = {
    'F6P': p[3],
    'F16P': p[5],
    'GAP': p[7],
    'DHAP': p[9],
    '13DPG': p[11],
    '3PG': p[13],
    '2PG': p[15],
    'PEP': p[17]
}
default_initial_conditions = {k:G_to_K(x,RT_kcal) for k,x in concentration_potentials.items()}

static_concentration_potentials = {
    'G6P': p[1],
    'PYR': p[19],
    'NADH': p[21],
    'NAD': p[23],
    'ATP': p[25],
    'ADP': p[27],
    'AMP': p[29],
    'Pi': p[31],
    'H2O': p[33],
    'H': p[35]
}
default_static_concentrations = {k:G_to_K(x,RT_kcal) for k,x in static_concentration_potentials.items()}


class GlycolysisGK(BondGraphBioChem):
    """
    The model of glycolysis defined in Mason and Covert (2019)
    """
    def __init__(self,name="GlycolysisGK", fitted_params=p):
        BondGraph.__init__(self,name=name)
        reactions, anchors = initialise_glycolysis_scaffold(self)
        self.reactions = reactions
        GK_laws = GK_components()
        bgt.add(self,GK_laws)
        connect_anchors_to_reactions(GK_laws,reactions,anchors)
        
        self.RT_kcal = RT_kcal
        
        self.R = R
        self.T = 310
        self.fitted_params = fitted_params
        
        # Set default parameter values
        self._dynamic_species_parameters()
        self._chemostat_parameters()
        self._reaction_rate_parameters()
        self._enzyme_concentrations()
        self._binding_parameters()
        self.default_params = self.simulation_cvs
    
    def _dynamic_species_parameters(self):
        p = self.fitted_params
        species_standard_potentials = {
            'F6P': p[2],
            'F16P': p[4],
            'GAP': p[6],
            'DHAP': p[8],
            '13DPG': p[10],
            '3PG': p[12],
            '2PG': p[14],
            'PEP': p[16]
        }
        self.dynamic_species_parameters = {
            m:G_to_K(x,self.RT_kcal) for m,x in species_standard_potentials.items()
        }
        return self.dynamic_species_parameters

    def _chemostat_parameters(self):
        p = self.fitted_params
        RT_kcal = self.RT_kcal
        self.chemostat_parameters = {
            'G6P': (p[0]+p[1])/RT_kcal,
            'PYR': (p[18]+p[19])/RT_kcal,
            'NADH': (p[20]+p[21])/RT_kcal,
            'NAD': (p[22]+p[23])/RT_kcal,
            'ATP': (p[24]+p[25])/RT_kcal,
            'ADP': (p[26]+p[27])/RT_kcal,
            'AMP': (p[28]+p[29])/RT_kcal,
            'Pi': (p[30]+p[31])/RT_kcal,
            'H2O': (p[32]+p[33])/RT_kcal,
            'H': (p[34]+p[35])/RT_kcal
        }

    def _reaction_rate_parameters(self):
        p = self.fitted_params
        reaction_affinities = {
            'pgi': -p[36],
            'pfk': -p[38],
            'fbp': -p[40],
            'fba': -p[42],
            'tpi': -p[44],
            'gap': -p[46],
            'pgk': -p[48],
            'gpm': -p[50],
            'eno': -p[52],
            'pyk': -p[54],
            'pps': -p[56]
        }
        self.reaction_rate_parameters = {r:G_to_K(x,self.RT_kcal) for r,x in reaction_affinities.items()}
        return self.reaction_rate_parameters
    
    def _enzyme_concentrations(self):
        p = self.fitted_params
        enzyme_potentials = {
            'pgi': p[37],
            'pfk': p[39],
            'fbp': p[41],
            'fba': p[43],
            'tpi': p[45],
            'gap': p[47],
            'pgk': p[49],
            'gpm': p[51],
            'eno': p[53],
            'pyk': p[55],
            'pps': p[57]
        }
        self.enzyme_concentrations = {e:G_to_K(x,self.RT_kcal) for e,x in enzyme_potentials.items()}
        return self.enzyme_concentrations
    
    def _binding_parameters(self):
        p = self.fitted_params
        binding_potentials = {
            'pgi': {
                "Rb0": ("G6P", p[58]+p[0]),
                "Rb1": ("F6P", p[78]+p[2]),
            },
            'pfk': {
                "RbA": ("F6P", p[59]+p[2]),
                "RbB": ("ATP", p[60]+p[24]),
                "RbX": ("F16P", p[81]+p[4]),
                "RbY": ("H", p[79]+p[34]),
                "RbZ": ("ADP", p[80]+p[26]),
            },
            'fbp': {
                "RbA": ("F16P", p[61]+p[4]),
                "RbB": ("H2O", p[62]+p[32]),
                "RbX": ("F6P", p[83]+p[2]),
                "RbY": ("Pi", p[82]+p[30])
            },
            'fba': {
                "RbA": ("F16P", p[63]+p[4]),
                "RbX": ("GAP", p[85]+p[6]),
                "RbY": ("DHAP", p[84]+p[8])
            },
            'tpi': {
                "Rb0": ("DHAP", p[64]+p[8]),
                "Rb1": ("GAP", p[86]+p[6]),
            },
            'gap': {
                "RbA": ("GAP", p[65]+p[6]),
                "RbB": ("NAD", p[66]+p[22]),
                "RbC": ("Pi", p[67]+p[30]),
                "RbX": ("13DPG", p[89]+p[10]),
                "RbY": ("NADH", p[87]+p[20]),
                "RbZ": ("H", p[88]+p[34]),
            },
            'pgk': {
                "RbA": ("13DPG", p[68]+p[10]),
                "RbB": ("ADP", p[69]+p[26]),
                "RbX": ("3PG", p[91]+p[12]),
                "RbY": ("ATP", p[90]+p[24])
            },
            'gpm': {
                "Rb0": ("3PG", p[70]+p[12]),
                "Rb1": ("2PG", p[92]+p[14]),
            },
            'eno': {
                "RbA": ("2PG", p[71]+p[14]),
                "RbX": ("PEP", p[94]+p[16]),
                "RbY": ("H2O", p[93]+p[32])
            },
            'pyk': {
                "RbA": ("PEP", p[72]+p[16]),
                "RbB": ("ADP", p[73]+p[26]),
                "RbC": ("H", p[74]+p[34]),
                "RbX": ("PYR", p[96]+p[18]),
                "RbY": ("ATP", p[95]+p[24]),
            },
            'pps': {
                "RbA": ("PYR", p[75]+p[18]),
                "RbB": ("ATP", p[76]+p[24]),
                "RbC": ("H2O", p[77]+p[32]),
                "RbW1": ("H", p[99]+p[34]),
                "RbW2": ("H", p[100]+p[34]),
                "RbX": ("PEP", p[101]+p[16]),
                "RbY": ("AMP", p[97]+p[28]),
                "RbZ": ("Pi", p[98]+p[30])
            }
        }
        self.binding_parameters = {
            r:{
                m:(x[0],G_to_K(x[1],self.RT_kcal)) for m,x in d.items()
            } for r,d in binding_potentials.items()
        }
        return self.binding_parameters
        
    @property
    def simulation_cvs(self):
        species_params = self.dynamic_species_parameters
        chemostat_params = self.chemostat_parameters
        rate_params = self.reaction_rate_parameters
        enzyme_conc = self.enzyme_concentrations
        binding_constants = self.binding_parameters

        parameter_mappings = {}
        for k,v in species_params.items():
            parameter_mappings[(self/f"C:{k}", "k")] = v
        for k,v in chemostat_params.items():
            parameter_mappings[(self/f"SS:{k}", "e")] = v
        for k in rate_params.keys():
            parameter_mappings[(self/f"R:{k}", "r")] = rate_params[k]
            parameter_mappings[(self/f"R:{k}", "e_T")] = enzyme_conc[k]
            for pname,v in binding_constants[k].items():
                parameter_mappings[(self/f"R:{k}", pname)] = v[1]
        
        return parameter_mappings

class GlycolysisGK_Static(StaticBGBioChem,GlycolysisGK):
    def __init__(self,name="GlycolysisGK", fitted_params=p):
        GlycolysisGK.__init__(self,name,fitted_params)

def GK_components():
    """
    The reaction components for the generalised kinetics model
    """
    return [
        bgt.new(component="R_MM",name='pgi',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="GK_ABXYZ",name='pfk',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="GK_ABXY",name='fbp',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="GK_AXY",name='fba',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="R_MM",name='tpi',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="GK_ABCXYZ",name='gap',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="GK_ABXY",name='pgk',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="R_MM",name='gpm',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="GK_AXY",name='eno',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="GK_ABCXY",name='pyk',library="Enzyme",value={'R':1,'T':1}),
        bgt.new(component="GK_ABCWWXYZ",name='pps',library="Enzyme",value={'R':1,'T':1}),
    ]

# Concentrations obtained from simulating the generalised kinetics model to steady state
x_ss = np.array([
    0.0003527435876923483, # F6P
    0.0011272144922777165, # F16P
    0.0008311267493691017, # GAP
    0.000905955103763534, # DHAP
    0.0008005216906260211, # 13DPG
    0.0008151225382543824, # 3PG
    0.0002474937759540567, # 2PG
    0.00010868565709226159 # PEP
])

class GlycolysisMM(BondGraphBioChem):
    """
    A model of glycolysis using Michaelis-Menten kinetics, simplified from the full generalised kinetics model
    """
    def __init__(self,name="GlycolysisMM"):
        BondGraph.__init__(self,name=name)
        reactions, anchors = initialise_glycolysis_scaffold(self)
        self.reactions = reactions
        MM_comps = MM_components()
        bgt.add(self,MM_comps)
        connect_anchors_to_reactions(MM_comps,reactions,anchors)
        
        self.R = R
        self.T = 310
        
        self.calculate_parameters()
        self.default_params = self.simulation_cvs
        
    def static_reactants(self,reaction):
        """
        Extracts "side" reactants of a reaction
        """
        r_dict = {r[0]:r[1] for r in self.reactions}
        return [reactant for reactant in r_dict[reaction] if reactant in self.pathway_metabolites]
    
    def static_products(self,reaction):
        """
        Extracts "side" products of a reaction
        """
        r_dict = {r[0]:r[2] for r in self.reactions}
        products = [extract_stoichiometry(product)[1] for product in r_dict[reaction]]
        return [x for x in products if x in self.pathway_metabolites]
    
    def primary_reactant(self,reaction):
        r_dict = {r[0]:r[1] for r in self.reactions}
        return r_dict[reaction][0]
    
    def primary_product(self,reaction):
        r_dict = {r[0]:r[2] for r in self.reactions}
        if reaction == "pps":
            return r_dict[reaction][1]
        else:
            return r_dict[reaction][0]
    
    @property
    def static_metabolites(self):
        return [c.name for c in self.components if c.metamodel=="SS"]
    
    @property
    def pathway_metabolites(self):
        """
        Extracts the dynamic species and the upstream and downstream species of the glycolysis pathway
        """
        return [x for x in self.static_metabolites if x not in ["G6P","PYR"]]
    
    def adjustment_constants(self,reaction,direction,GK_binding_constants):
        """
        Calculates the constants required to adjust binding constants for Michaelis-Menten kinetics.
        """
        if direction == "f":
            metabolites = self.static_reactants(reaction)
        elif direction == "r":
            metabolites = self.static_products(reaction)
            
        binding_adjustment = 1
        side_potentials = 0
        
        GK_parameters = GK_binding_constants[reaction]
        for _,(rname,v) in GK_parameters.items():
            if rname in metabolites:
                chemical_potential = self.chemostat_parameters[rname]
                binding_adjustment *= (1 + np.exp(chemical_potential) / v)
                side_potentials += chemical_potential
        return binding_adjustment,side_potentials
    
    def simplified_reaction_parameters(self,reaction,GK_rate_params,GK_binding_constants):
        (Zf,Af_side) = self.adjustment_constants(reaction,"f",GK_binding_constants)
        (Zr,Ar_side) = self.adjustment_constants(reaction,"r",GK_binding_constants)
        
        adapted_binding_constants = {v1:v2 for v1,v2 in GK_binding_constants[reaction].values()}
        
        rate_parameter = GK_rate_params[reaction]/(Zf+Zr-1)
        
        base_reactant = self.primary_reactant(reaction)
        original_parameter = adapted_binding_constants[base_reactant]
        forward_binding_parameter = original_parameter*(Zf+Zr-1)/Zf/np.exp(-Af_side)
        
        base_product = self.primary_product(reaction)
        original_parameter = adapted_binding_constants[base_product]
        reverse_binding_parameter = original_parameter*(Zf+Zr-1)/Zr/np.exp(-Ar_side)
        
        return rate_parameter,forward_binding_parameter,reverse_binding_parameter

    def calculate_parameters(self):
        """
        Calculates parameters for approximate Michaelis-Menten kinetics from generalised kinetics parameters
        """
        GK_model = GlycolysisGK()
        self.dynamic_species_parameters = GK_model.dynamic_species_parameters
        self.chemostat_parameters = GK_model.chemostat_parameters
        self.enzyme_concentrations = GK_model.enzyme_concentrations
        
        GK_rate_params = GK_model.reaction_rate_parameters
        GK_binding_constants = GK_model.binding_parameters
        
        rate_parameters = {}
        binding_parameters = {}
        for reaction in self.reactions:
            rname = reaction[0]
            if not rname == "fba":
                r,Rb0,Rb1 = self.simplified_reaction_parameters(rname,GK_rate_params,GK_binding_constants)
                rate_parameters[rname] = r
                binding_parameters[rname] = {"Rb0":Rb0, "Rb1":Rb1}
                
        # Need different equations to fit fba kinetics
        fba = 'fba'
        rate_parameters[fba] = GK_rate_params[fba]
        Rb0 = GK_binding_constants[fba]['RbA'][1]
        v_ss = compute_flux(GK_model,fba,x_ss)
        rate_term = rate_parameters[fba] * self.enzyme_concentrations[fba]
        fwd_ma_term,rev_ma_term = split_ma_terms(self,fba,x_ss)
        ma_term = fwd_ma_term - rev_ma_term
        Rb1 = rev_ma_term/(rate_term*ma_term/v_ss - fwd_ma_term/Rb0 - 1)
        binding_parameters[fba] = {'Rb0':Rb0, 'Rb1':Rb1}
        
        self.reaction_rate_parameters = rate_parameters
        self.binding_parameters = binding_parameters
    
    @property
    def simulation_cvs(self):
        parameter_mappings = {}
        for k,v in self.dynamic_species_parameters.items():
            parameter_mappings[(self/f"C:{k}", "k")] = v
        for k,v in self.chemostat_parameters.items():
            parameter_mappings[(self/f"SS:{k}", "e")] = v
        for k in self.reaction_rate_parameters.keys():
            parameter_mappings[(self/k/f"R:{k}", "r")] = self.reaction_rate_parameters[k]
            parameter_mappings[(self/k/f"R:{k}", "e_T")] = self.enzyme_concentrations[k]
            for pname,v in self.binding_parameters[k].items():
                parameter_mappings[(self/k/f"R:{k}", pname)] = v

        return parameter_mappings

class GlycolysisMM_Static(StaticBGBioChem,GlycolysisMM):
    def __init__(self,name="GlycolysisMM"):
        GlycolysisMM.__init__(self,name)

class GlycolysisMA(BondGraphBioChem):
    def __init__(self,name="GlycolysisMA"):
        BondGraph.__init__(self,name=name)
        reactions, anchors = initialise_glycolysis_scaffold(self)
        self.reactions = reactions
        MA_comps = MA_components()
        bgt.add(self,MA_comps)
        connect_anchors_to_reactions(MA_comps,reactions,anchors)
        
        self.R = R
        self.T = 310
        
        self.calculate_parameters()
        self.default_params = self.simulation_cvs

    def calculate_parameters(self):
        GK_model = GlycolysisGK()
        self.dynamic_species_parameters = GK_model.dynamic_species_parameters
        self.chemostat_parameters = GK_model.chemostat_parameters
        self.enzyme_concentrations = GK_model.enzyme_concentrations
        
        # Set rate parameters to match steady-state fluxes
        rate_parameters = {}
        for r,_,_ in self.reactions:
            v_ss = compute_flux(GK_model,r,x_ss)
            ma_term = mass_action_term(GK_model,r,x_ss)
            rate_parameters[r] = v_ss/ma_term
        
        self.reaction_rate_parameters = rate_parameters
    
    @property
    def simulation_cvs(self):
        parameter_mappings = {}
        for k,v in self.dynamic_species_parameters.items():
            parameter_mappings[(self/f"C:{k}", "k")] = v
        for k,v in self.chemostat_parameters.items():
            parameter_mappings[(self/f"SS:{k}", "e")] = v
        for k in self.reaction_rate_parameters.keys():
            parameter_mappings[(self/k/f"R:{k}", "r")] = self.reaction_rate_parameters[k]

        return parameter_mappings

class GlycolysisMA_Static(StaticBGBioChem,GlycolysisMA):
    def __init__(self,name="GlycolysisMA"):
        GlycolysisMA.__init__(self,name)

glycolysis_reaction_structure = [
    ('pgi',1,1),
    ('pfk',2,3),
    ('fbp',2,2),
    ('fba',1,2),
    ('tpi',1,1),
    ('gap',3,3),
    ('pgk',2,2),
    ('gpm',1,1),
    ('eno',1,2),
    ('pyk',3,2),
    ('pps',3,4)
]

def MA_components(reactions=glycolysis_reaction_structure):
    """
    Returns modules containing both the reaction and flow junction components for the mass action model
    """
    return [reaction_structure(r,p,Re(name=n,value={'R':1,'T':1})) 
            for n,r,p in reactions]

def MM_components(reactions=glycolysis_reaction_structure):
    """
    Returns modules containing both the reaction and flow junction components for the Michaelis-Menten model
    """
    return [reaction_structure(r,p,bgt.new(component="R_MM",name=n,library="Enzyme",value={'R':1,'T':1})) 
            for n,r,p in reactions]

def MM_flux(model,reaction,x):
    """
    Calculates the flux of a reaction in the Michaelis-Menten model
    """
    binding_constants = model.binding_parameters[reaction]
    
    rate_term = model.reaction_rate_parameters[reaction] * model.enzyme_concentrations[reaction]
    ma_term = mass_action_term(model,reaction,x)
    
    fwd_ma_term,rev_ma_term = split_ma_terms(model,reaction,x)
    reg_term = 1 + fwd_ma_term/binding_constants['Rb0'] + rev_ma_term/binding_constants['Rb1']
    
    return rate_term * ma_term / reg_term

def initialise_glycolysis_scaffold(model):
    """
    Adds a glycolysis scaffold to the model in the arguments, with the species and potential junctions,
    but not the reaction components.
    """
    static_metabolites = ['G6P','PYR','NADH','NAD','ATP','ADP','AMP','Pi','H2O','H']
    dynamic_metabolites = ['F6P','F16P','GAP','DHAP','13DPG','3PG','2PG','PEP']
    reactions = [
        ('pgi',('G6P',),('F6P',)),
        ('pfk',('F6P','ATP'),('F16P','H','ADP')),
        ('fbp',('F16P','H2O'),('F6P','Pi')),
        ('fba',('F16P',),('GAP','DHAP')),
        ('tpi',('DHAP',),('GAP',)),
        ('gap',('GAP','NAD','Pi'),('13DPG','NADH','H')),
        ('pgk',('13DPG','ADP'),('3PG','ATP')),
        ('gpm',('3PG',),('2PG',)),
        ('eno',('2PG',),('PEP','H2O')),
        ('pyk',('PEP','ADP','H'),('PYR','ATP')),
        ('pps',('PYR','ATP','H2O'),('2*H','PEP','AMP','Pi'))
    ]

    Se_components = [Se(name=x) for x in static_metabolites]
    Ce_components = [Ce(name=x,value={'R':1,'T':1}) for x in dynamic_metabolites]
    bgt.add(model,Se_components)
    bgt.add(model,Ce_components)

    anchors = {}
    for c in Se_components+Ce_components:
        new_component = bgt.new('0',name=c.name)
        anchors[c.name] = new_component
        bgt.add(model,new_component)
        bgt.connect(new_component,c)

    # Add a separate anchor for H with a stoichiometry of 2
    TF = bgt.new('TF',name='pps-H',value={'r':2})
    bgt.add(model,TF)
    bgt.connect(anchors['H'],(TF,0))
    anchors['2*H'] = (TF,1)
    
    return reactions, anchors

def connect_anchors_to_reactions(reaction_comps,reactions,anchors):
    """
    Connects the anchors (0 junctions corresponding to species) to reactions
    """
    comp_dict = {c.name: c for c in reaction_comps}
    for name,reactants,products in reactions:
        port = 0
        for r in reactants:
            anchor = anchors[r]
            bgt.connect(anchor,(comp_dict[name],port))
            port += 1
        for p in products:
            anchor = anchors[p]
            bgt.connect((comp_dict[name],port),anchor)
            port += 1

def extract_stoichiometry(s):
    if "*" in s:
        x = s.split("*")
        x[0] = int(x[0])
        return x
    else:
        return [1,s]

def compute_flux(model,reaction,x):
    """
    Calculates the flux of a reaction in the generalised kinetics model
    """
    rate_term = model.reaction_rate_parameters[reaction] * model.enzyme_concentrations[reaction]
    ma_term = mass_action_term(model,reaction,x)
    reg_term = regulation_term(model,reaction,x)
    return rate_term * ma_term / reg_term

def split_ma_terms(model,reaction,x):
    """
    Calculates the forward and reverse affinities for a reaction.
    If the species is dynamic, Kx is used.
    If the species is static, exp(mu/RT) is used.
    """
    concentrations = {s.name:c for (s,t),c in zip(model.state_vars.values(),x)}
    
    reactant_dict = {r[0]:tuple(extract_stoichiometry(x) for x in r[1]) for r in model.reactions}
    forward_term = np.prod([_species_lookup(model,s,concentrations)**n for n,s in reactant_dict[reaction]])
            
    product_dict = {r[0]:tuple(extract_stoichiometry(x) for x in r[2]) for r in model.reactions}
    reverse_term = np.prod([_species_lookup(model,s,concentrations)**n for n,s in product_dict[reaction]])
    
    return forward_term,reverse_term

def mass_action_term(model,reaction,x):
    """
    Returns exp(Af/RT) - exp(Ar/RT)
    """
    forward_term,reverse_term = split_ma_terms(model,reaction,x)
    return forward_term - reverse_term

def regulation_term(model,reaction,x):
    """
    Returns the denominator of the generalised kinetics rate law.
    """
    concentrations = {s.name:c for (s,t),c in zip(model.state_vars.values(),x)}
    binding_constants = model.binding_parameters[reaction]

    reactant_dict = {r[0]:tuple(extract_stoichiometry(x)[1] for x in r[1]) for r in model.reactions}
    reactants = reactant_dict[reaction]
    
    forward_term = 1
    reverse_term = 1
    for s,v in binding_constants.values():
        if s in reactants:
            forward_term *= (1 + _species_lookup(model,s,concentrations)/v)
        else:
            reverse_term *= (1 + _species_lookup(model,s,concentrations)/v)
    return forward_term + reverse_term - 1

def _species_lookup(model,species,concentrations):
    """
    Returns Kx if species is dynamic, and exp(mu/RT) if the species is static
    """
    if species in concentrations.keys():
        return (model.dynamic_species_parameters[species]*concentrations[species])
    else:
        return np.exp(model.chemostat_parameters[species])

def reaction_structure(n_reactants, n_products, reaction_component):
    """
    Builds a module for a reaction with a given number of reactants and products.
    Bonds that would connect to species have are exposed.
    """
    module = BondGraphBioChem(name=reaction_component.name)
    bgt.add(module,reaction_component)
    
    if n_reactants > 1:
        f_complex = bgt.new("1",name="FComplex")
        bgt.add(module,f_complex)
        bgt.connect(f_complex,(reaction_component,0))
    else:
        f_complex = (reaction_component,0)
        
    if n_products > 1:
        r_complex = bgt.new("1",name="RComplex")
        bgt.add(module,r_complex)
        bgt.connect((reaction_component,1),r_complex)
    else:
        r_complex = (reaction_component,1)
    
    for i in range(n_reactants):
        port = bgt.new("SS",name=f"reac{i}")
        bgt.add(module,port)
        bgex.expose(port,label=f"s{i}")
        bgt.connect(port,f_complex)
    
    for i in range(n_products):
        port = bgt.new("SS",name=f"prod{i}")
        bgt.add(module,port)
        bgex.expose(port,label=f"s{n_reactants+i}")
        bgt.connect(r_complex,port)
    
    return module

def perturb_internal_species(model,perturbation,
                             x0_base=deepcopy(x_ss),
                             tmax=300.0):
    dynamic_metabolites = ['F6P','F16P','GAP','DHAP','13DPG','3PG','2PG','PEP']
    simulation_results = {}
    tspan = (0.0,tmax)
    for i,s in enumerate(dynamic_metabolites):
        x0 = deepcopy(x0_base)
        x0[i] *= (1 + perturbation)
        cv = bgex.gather_cv(model)
        sol = simulate(model,tspan,x0,control_vars=cv)
        simulation_results[s] = sol
    return simulation_results

def relaxation_time(sol,x_ss=x_ss):
    diff = sol.u - x_ss
    distance = [np.linalg.norm(v) for v in diff]
    max_dist = np.max(distance)
    threshold = 0.05*max_dist
    i = np.max(np.where(distance > threshold))
    t0 = sol.t[i]
    t1 = sol.t[i+1]
    d0 = distance[i]
    d1 = distance[i+1]
    return t0+(t1-t0)*(threshold-d0)/(d1-d0)

def all_relaxation_times(sim_results,x_ss=x_ss):
    result = {}
    for s,sol in sim_results.items():
        if x_ss is None:
            diff = sol.u - sol.u[-1]
        else:
            diff = sol.u - x_ss
        distance = [np.linalg.norm(v) for v in diff]
        max_dist = np.max(distance)
        threshold = 0.05*max_dist
        i = np.max(np.where(distance > threshold))
        t0 = sol.t[i]
        t1 = sol.t[i+1]
        d0 = distance[i]
        d1 = distance[i+1]
        relaxation_time = t0+(t1-t0)*(threshold-d0)/(d1-d0)
        result[s] = relaxation_time
    return result

def internal_perturbation(GK_model,MM_model,MA_model,perturbation):
    GK_simulation_results = perturb_internal_species(GK_model,perturbation)
    MM_simulation_results = perturb_internal_species(MM_model,perturbation)
    MA_simulation_results = perturb_internal_species(MA_model,perturbation)
    GK_relax_time = all_relaxation_times(GK_simulation_results)
    MM_relax_time = all_relaxation_times(MM_simulation_results)
    MA_relax_time = all_relaxation_times(MA_simulation_results)
    return ((GK_simulation_results,MM_simulation_results,MA_simulation_results),
            (GK_relax_time,MM_relax_time,MA_relax_time))

def plot_internal_perturbation(sim_results,relaxation_times):
    (GK_simulation_results,MM_simulation_results,MA_simulation_results) = sim_results
    (GK_relax_time,MM_relax_time,MA_relax_time) = relaxation_times
    fig,ax = plt.subplots(2,len(GK_relax_time),figsize=(18,4.5))
    width = 0.9
    for i,s in enumerate(GK_relax_time.keys()):
        x = np.arange(3)
        a = ax[0,i]
        a.set_title(s)
        a.bar(0,GK_relax_time[s],width,label="GK")
        a.bar(1,MM_relax_time[s],width,label="MM")
        a.bar(2,MA_relax_time[s],width,label="MA")
        a.set_xticks([])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)

    for i,s in enumerate(GK_simulation_results.keys()):
        a = ax[1,i]
        tmax = min(300,1.5*GK_relax_time[s])
        t = np.linspace(0,tmax,200)
        sol = GK_simulation_results[s]
        a.plot(t,sol(t)[-1],label="GK")
        sol = MM_simulation_results[s]
        a.plot(t,sol(t)[-1],label="MM")
        sol = MA_simulation_results[s]
        a.plot(t,sol(t)[-1],label="MA")
        a.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        a.set_xlabel("Time (s)")
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        
    ax[1,-1].legend()
    ax[0,0].set_ylabel("Response time (s)")
    ax[1,0].set_ylabel("[PEP] (M)")
    return fig,ax

def perturb_external_species(model,perturbation,
                             x0=deepcopy(x_ss),
                             tmax=1000.0):
    tspan = (0.0,tmax)
    base_chemostats = deepcopy(model.chemostat_parameters)
    # Omit H2O as it is the solvent. NADH omitted as perturbing it causes little change.
    perturbed_species = [k for k in base_chemostats.keys() if k not in ["H2O","NADH"]]
    simulation_results = {}
    
    for k in perturbed_species:
        chemostat_parameters = deepcopy(base_chemostats)
        chemostat_parameters[k] += np.log(1 + perturbation)
        model.chemostat_parameters = chemostat_parameters

        cv = bgex.gather_cv(model)
        sol = simulate(model,tspan,x0,control_vars=cv,abstol=1e-10,reltol=1e-10)
        simulation_results[k] = sol

    model.chemostat_parameters = base_chemostats
    return simulation_results

def steady_state_deviations(sim_results,x_ss=x_ss):
    errors = {}
    for s in sim_results.keys():
        sol = sim_results[s]
        errors[s] = np.linalg.norm(sol.u[-1]-x_ss)
    return errors

def steady_state_errors(sim_results1,sim_results2):
    errors = {}
    for s in sim_results1.keys():
        sol1 = sim_results1[s]
        sol2 = sim_results2[s]
        errors[s] = np.linalg.norm(sol1.u[-1]-sol2.u[-1])
    return errors

def external_perturbations(GK_model,MM_model,MA_model,perturbation):
    GK_simulation_results = perturb_external_species(GK_model,perturbation)
    MM_simulation_results = perturb_external_species(MM_model,perturbation)
    MA_simulation_results = perturb_external_species(MA_model,perturbation)
    sim_results = (GK_simulation_results,MM_simulation_results,MA_simulation_results)
    
    GK_relax_time = all_relaxation_times(GK_simulation_results,None)
    MM_relax_time = all_relaxation_times(MM_simulation_results,None)
    MA_relax_time = all_relaxation_times(MA_simulation_results,None)
    relax_times = (GK_relax_time,MM_relax_time,MA_relax_time)

    GK_dev = steady_state_deviations(GK_simulation_results)
    MM_dev = steady_state_deviations(MM_simulation_results)
    MA_dev = steady_state_deviations(MA_simulation_results)
    ss_dev = (GK_dev,MM_dev,MA_dev)
    
    return sim_results,relax_times,ss_dev

def plot_external_perturbation(sim_results,relax_times,ss_dev):
    GK_simulation_results,MM_simulation_results,MA_simulation_results = sim_results
    GK_relax_time,MM_relax_time,MA_relax_time = relax_times
    GK_dev,MM_dev,MA_dev = ss_dev
    
    n_ex = len(GK_simulation_results.keys())
    species = [s for s in GK_relax_time.keys()]

    fig,ax = plt.subplots(3,len(species),figsize=(20,6.75))
    width = 0.9
    for i,s in enumerate(species):
        x = np.arange(3)
        a = ax[0,i]
        a.set_title(s)
        plot_relaxation_bar(a,0,GK_relax_time[s],width,label="GK")
        plot_relaxation_bar(a,1,MM_relax_time[s],width,label="MM")
        plot_relaxation_bar(a,2,MA_relax_time[s],width,label="MA")
        a.set_xticks([])
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)

    for i,s in enumerate(species):
        a = ax[1,i]
        a.bar(0,GK_dev[s],width,label="GK")
        a.bar(1,MM_dev[s],width,label="MM")
        a.bar(2,MA_dev[s],width,label="MA")
        a.set_xticks([])
        a.ticklabel_format(axis="y",style='sci',scilimits=(0,0))
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        
    for i,s in enumerate(species):
        a = ax[2,i]
        tmax = min(1000,1.5*GK_relax_time[s])
        t = np.linspace(0,tmax,200)
        sol = GK_simulation_results[s]
        a.plot(t,sol(t)[-1],label="GK")
        sol = MM_simulation_results[s]
        a.plot(t,sol(t)[-1],label="MM")
        sol = MA_simulation_results[s]
        a.plot(t,sol(t)[-1],label="MA")
        a.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        a.set_xlabel("Time (s)")
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        
    ax[2,-1].legend()
    ax[0,0].set_ylabel("Response time (s)")
    ax[1,0].set_ylabel("Steady state change")
    ax[2,0].set_ylabel("[PEP] (M)")
        
    return fig,ax

def annotate(ax,rects,label):
    """Attach a text label above each bar in *rects*."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -2), 
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def plot_relaxation_bar(a,x,relax_time,width,label):
    rect = a.bar(x,min(relax_time,500),width,label=label)
    if relax_time > 500:
        annotate(a,rect,">500")