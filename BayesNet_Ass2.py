# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 00:32:51 2022

@author: Jake
"""
# Necessary libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import copy
import time

# combinatorics
from itertools import product, combinations

from DiscreteFactors_Ass2 import Factor
from Graph_Ass2 import Graph

def allEqualThisIndex(dict_of_arrays, **fixed_vars):
    """
    Helper function to create a boolean index vector into a tabular data structure,
    such that we return True only for rows of the table where, e.g.
    column_a=fixed_vars['column_a'] and column_b=fixed_vars['column_b'].
    
    This is a simple task, but it's not *quite* obvious
    for various obscure technical reasons.
    
    It is perhaps best explained by an example.
    
    >>> all_equal_this_index(
    ...    {'X': [1, 1, 0], Y: [1, 0, 1]},
    ...    X=1,
    ...    Y=1
    ... )
    [True, False, False]
    """
    # base index is a boolean vector, everywhere true
    first_array = dict_of_arrays[list(dict_of_arrays.keys())[0]]
    index = np.ones_like(first_array, dtype=np.bool_)
    for var_name, var_val in fixed_vars.items():
        index = index & (np.asarray(dict_of_arrays[var_name])==var_val)
    return index

def estimateFactor(data, var_name, parent_names, outcomeSpace, alpha = 0):
    """
    Calculate a dictionary probability table by ML given
    `data`, a dictionary or dataframe of observations
    `var_name`, the column of the data to be used for the conditioned variable and
    `parent_names`, a tuple of columns to be used for the parents and
    `outcomeSpace`, a dict that maps variable names to a tuple of possible outcomes
    Return a dictionary containing an estimated conditional probability table.
    """    
    var_outcomes = outcomeSpace[var_name]
    parent_outcomes = [outcomeSpace[var] for var in (parent_names)]
    # cartesian product to generate a table of all possible outcomes
    all_parent_combinations = product(*parent_outcomes)

    f = Factor(list(parent_names)+[var_name], outcomeSpace)
    
    for i, parent_combination in enumerate(all_parent_combinations):
        parent_vars = dict(zip(parent_names, parent_combination))
        parent_index = allEqualThisIndex(data, **parent_vars)
        for var_outcome in var_outcomes:
            var_index = (np.asarray(data[var_name])==var_outcome)
            f[tuple(list(parent_combination)+[var_outcome])] = ((var_index & parent_index).sum()+alpha)/(parent_index.sum()+alpha*len(var_outcomes))
            
    return f

class BayesNet():
    def __init__(self, graph, outcomeSpace=None, factor_dict=None):
        self.graph = graph
        self.outcomeSpace = dict()
        self.factors = dict()
        if outcomeSpace is not None:
            self.outcomeSpace = outcomeSpace
        if factor_dict is not None:
            self.factors = factor_dict
            
    def learnParameters(self, data, alpha = 0):
        '''
        Iterate over each node in the graph, and use the given data
        to estimate the factor P(node|parents), then add the new factor 
        to the `self.factors` dictionary.
        '''
        graphT = self.graph.transpose()
        for node, parents in graphT.adj_list.items():
            data = data[pd.isnull(data[node]) == False]
            self.factors[node] = estimateFactor(data, node, parents, self.outcomeSpace, alpha = alpha) 
            
    def joint(self):
        '''
        Join every factor in the network, and return the resulting factor.
        '''
        factor_list = list(self.factors.values())
        
        accumulator = factor_list[0]
        for factor in factor_list[1:]:
            accumulator = accumulator*factor
        
        return accumulator
    
    def inferenceByEnum(self, order, counts = True):
        if counts:
            muls = 0
            adds = 0
        # Let's make a list of factors
        factorList = self.factors.values()
        
        # Create an empty factor as accumulator 
        newFactor = Factor(tuple(), self.outcomeSpace)

        first = True
        # Lets iterate over all factors
        for f in factorList:
            newFactor = newFactor*f
            if not first and counts:
                muls += newFactor.table.size # keep track of multiplications being done with each join operation
            first = False
            
        for var in order:
            # Now, we need to remove var from the domain of the new factor.
            if counts:
                adds += newFactor.table.size # keep track of the number of additions being done in the following marginalization
            newFactor = newFactor.marginalize(var) 

        if counts:
            print('Amount of Multiplication Operations: ', muls)
            print('Amount of Addition Operations: ', adds)
            
        return newFactor
   
    def predict(self, class_var, evidence):
        '''
        Arguments:
            class_var:   Variable identifier to be classified
            evidence:    Python dictionary with one instantiation to all variables but class_var
        Returns:
            The MPE value (class label) of class_var given evidence
        ''' 
        
        # Creating a new factor to accumulate
        accumulator = self.factors[class_var].evidence(**evidence)
        
        # The only factors that impact the result are those that the class variable connects to (P(e|C)) and the class variable itself.
        
        for factor in self.graph.adj_list[class_var]:
            accumulator = accumulator*self.factors[factor].evidence(**evidence) 
       
        print(accumulator) 
       
        # This relies on the ordering in the table being the same as the outcomespace. I believe this is true based on how the tables are constructed but need to check.
        predicted_outcome = accumulator.outcomeSpace[class_var][np.argmax(accumulator.table)]

        return predicted_outcome 
    
    # Probability Prediction for Ass2, marginalises factors with missing variables.
    
    def predict_proba(self, class_var, evidence, missing = [], value = '0'):
        '''
        Arguments:
            class_var:   Variable identifier to be classified
            evidence:    Python dictionary with one instantiation to all variables but class_var
            missing:     List of missing variables to be marginalised, in elimination order
            value:       Value we want the probability for
        Returns:
            The class probabilties
        ''' 
        
        # Creating factor list of relevant factors - for naive bayes it is all factors except class factor as that is the beginning accumulation factor
        
        factorList = []
        
        for factor in self.factors:
            if factor != class_var:
                factorList.append(self.factors[factor])

        if len(missing) > 0: #If there is at least 1 variable that needs to be marginalised
    
            for var in missing:
                # We create an empty factor as an accumulator
                newFactor = Factor(tuple(), self.outcomeSpace)
                # A list to keep track of all the factors we will keep for the next step
                updatedFactorsList = list()            
    
                for f in factorList:
                    # and select the ones that have the variable to be eliminated
                    if var in f.domain:
                        # join the factor `f` with the accumulator `newFactor`
                        newFactor = newFactor*f
                            
                    else:
                        # since the factor `f` doesn't contain `var`, we will keep it for next iteration
                        updatedFactorsList.append(f)
            
                
                newFactor = newFactor.marginalize(var) 
                # append the new combined factor to the factor list
                updatedFactorsList.append(newFactor)
                # replace factorList with the updated factor list, ready for the next iteration
                factorList = updatedFactorsList
        
            
        accumulator = self.factors[class_var].evidence(**evidence)
        
        for factor in factorList:
            accumulator = accumulator*factor.evidence(**evidence) 
     
      
        # This relies on the ordering in the table being the same as the outcomespace. I believe this is true based on how the tables are constructed but need to check.
        
        
        index = accumulator.outcomeSpace[class_var].index(value)
        value_prob = accumulator.normalize().table[index]

        return value_prob
   
    def width(self, order):
        """
        argument 
        `order`, a list of variable names specifying an elimination order.

        Returns the width of the elimination order, i.e., the number of variables of the largest factor
        """   
        # Initialize w, a variable that has a width of the elimination order
        w = 0
        # Let's make a list of tuples, where each tuple is a factor domain. Create one tuple for each factor in the BN
        factorList = [factor.domain for factor in self.factors.values()]
        # We process the factor in elimination order
        for var in order:
            # This is the domain of the new factor. We use sets as it is handy to eliminate duplicate variables
            newFactorDom = set()
            # A list to keep track of all the factors we will keep for the next iteration (all factors not containing `var`)
            updatedFactorsList = list()            

            # Lets iterate over all factors
            for f_dom in factorList:
                # and select the ones that have the variable to be eliminated
                if var in f_dom:
                    # Merge the selected domain `f_dom` into the `newFactorDomain` set, simulating a join operation
                    newFactorDom.update(f_dom)
                else:
                    # since it doesn't contain `var`, we add the `var` factor to the updatedFactorsList to be processed in the next iteration
                    updatedFactorsList.append(f_dom)

            # Now, we need to remove var from the domain of the new factor. We are simulating a summation
            newFactorDom.remove(var)
            # Let's check if we have found a new largest factor
            if len(newFactorDom) > w:
                w = len(newFactorDom)
            # add the new combined factor domain to the list
            updatedFactorsList.append(newFactorDom)
            # replace factor list with updated factor list (getting rid of all factors containing var)
            factorList = updatedFactorsList

        return w
    
    def VE(self, order, counts = True):
        """
        argument 
        `order`, a list of variable names specifying an elimination order.

        Returns a single factor, the which remains after eliminating all other factors
        """   
        if counts:
            muls = 0
            adds = 0
        # Let's make a list of all factor objects
        factorList = self.factors.values()

        for var in order:
            # We create an empty factor as an accumulator
            newFactor = Factor(tuple(), self.outcomeSpace)
            first = True
            # A list to keep track of all the factors we will keep for the next step
            updatedFactorsList = list()            

            for f in factorList:
                # and select the ones that have the variable to be eliminated
                if var in f.domain:
                    # join the factor `f` with the accumulator `newFactor`
                    newFactor = newFactor*f
                    if not first and counts:
                        muls += newFactor.table.size # keep track of multiplications being done with each join operation
                    first = False
                        
                else:
                    # since the factor `f` doesn't contain `var`, we will keep it for next iteration
                    updatedFactorsList.append(f)
                    
            # Now, we need to remove var from the domain of the new factor. 
            if counts:
                adds += newFactor.table.size # keep track of the number of additions being done in the following marginalization
            newFactor = newFactor.marginalize(var) 
            # append the new combined factor to the factor list
            updatedFactorsList.append(newFactor)
            # replace factorList with the updated factor list, ready for the next iteration
            factorList = updatedFactorsList
        # for the final step, we join all remaining factors (usually there will only be one factor remaining)
        returnFactor = Factor(tuple(), self.outcomeSpace)
        for f in factorList:
            returnFactor = returnFactor*f
        if counts:
            print('Amount of Multiplication Operations: ', muls)
            print('Amount of Addition Operations: ', adds)
        return returnFactor
    
    def interactionGraph(self):
        '''
        Returns the interaction graph for this network.
        There are two ways to implement this function:
        - Iterate over factors and check which vars are in the same factors
        - Start with the directed graph, make it undirected and moralise it
        '''
        # Initialise an empty graph
        g = Graph()
        # Add each node to the graph
        for var in self.factors.keys():
            g.add_node(var)
        for factor in self.factors.values():
            # for every pair of vars in the domain
            for var1 in factor.domain:
                for var2 in factor.domain:
                    # check if connection already exists 
                    if var1 != var2 and var1 not in g.children(var2):
                        # Add undirected edge between nodes
                        g.add_edge(var1, var2, directed = False)

        return g
    
    def minDegree(self):
        # First get the interaction graph
        ig = self.interactionGraph()
        # Initialize order with empty list. This variable will have the answer in the end of the execution
        order = [] 
        # While the induced graph has nodes to be eliminated
        while len(ig) > 0:
            # Initialize minDegree with a large number: math.inf
            minDegree = math.inf
            for var in ig:
                # Test if var has a degree smaller than minDegree
                if len(ig.children(var)) < minDegree:
                    # We have found a new candidate to be the next eliminated variable. Let's save its degree and name
                    minDegree = len(ig.children(var))
                    minVar = var
            # We need to connect the neighbours of minVar, let us start using combinations function to find all pairs of minVar's neighbours
            for var1, var2 in combinations(ig.children(minVar), 2):
                # Check if these neighbour are not already connected by an edge
                if var1 not in ig.children(var2):
                    # add edge
                    ig.add_edge(var1,var2,directed = False)
            # Insert into order the variable in minVar
            order.append(minVar)
            # Now, we need to remove minVar from the adjacency list of every node
            ig.remove_node(minVar) 
        return order
        
    def query(self, q_vars, **q_evi):
        '''
        A faster VE-based query function
        Returns a factor P(q_vars| q_evi)
        '''
        # backup factors dict
        backup_factors = copy.deepcopy(self.factors)
        
        # Compute order through mindegree hueristic and remove query variables
        
        order = self.minDegree()
        order = [x for x in order if x not in q_vars]

        # Set evidence
        
        for factor in self.factors:
            self.factors[factor] = self.factors[factor].evidence(**q_evi)
            
        # Do variable elimination
        
        joint_marginal = self.VE(order = order)
        query_factor = joint_marginal.normalize()

        # restore original factors
        self.factors = backup_factors
        
        # Return factor P(q_vars | q_evi)
        return query_factor
    
class NaiveBayes(BayesNet):
    def predict_log(self, class_var, evidence):
        '''
        Arguments:
            class_var:   Variable identifier to be classified
            evidence:    Python dictionary with one instantiation to all variables but class_var
        Returns:
            The MPE value (class label) of class_var given evidence
        '''        
        # Backup factors as converting factors to log probabilities
        backup_factors = copy.deepcopy(self.factors)
        
        # Creating a new factor to accumulate
        accumulator = self.factors[class_var].evidence(**evidence).log_prob()
        
        # The only factors that impact the result are those that the class variable connects to (P(e|C)) and the class variable itself.      
        # Setting evidence for all relevent factors and converting to log probabilities
        
        for factor in self.graph.adj_list[class_var]:
            self.factors[factor].log_prob() # log_prob converts factor tables to log probabilities
            accumulator = accumulator.join(self.factors[factor].evidence(**evidence), log = True) # When log is true, it adds the tables rather than multiply     
        
        predicted_outcome = accumulator.outcomeSpace[class_var][np.argmax(accumulator.table)]
        self.factors = backup_factors
        
        return predicted_outcome

    
class HiddenMarkovModel():
    def __init__(self, start_state, transition, emission, variable_remap, outcomeSpace, transition_extra = None):
        '''
        Takes 3 arguments:
        - start_state: a factor representing the start state. E.g. domain might be ('A', 'B', 'C')
        - transition: a factor that represents the transition probs. E.g. P('A_next', 'B_next', 'C_next' | 'A', 'B', 'C')
        - emission: emission probabilities. E.g. P('O' | 'A', 'B', 'C')
        - variable_remap: a dictionary that maps new variable names to old variable names,
                            to reset the state after transition. E.g. {'A_next':'A', 'B_next':'B', 'C_next':'C'}
        '''
        self.state = start_state
        self.transition = transition
        self.emission = emission
        self.remap = variable_remap
        self.outcomeSpace = outcomeSpace
        self.transition_extra = transition_extra

        # These lists will be used later to find the mostly likely sequence of states
        self.history = []
        self.prev_history = []

    def forward(self, missing = [], value = '0', class_var = 'r1', **emission_evi):
    
        emissions = list(self.emission.values())
        transition_factor = self.transition
    
    
        if len(missing) > 0: #If there is at least 1 variable that needs to be marginalised
        
            for var in missing:
                if var in self.transition.domain:
                    transition_factor = transition_factor.marginalize(var)
                
                else:
                    # We create an empty factor as an accumulator
                    newFactor = Factor(tuple(), self.outcomeSpace)
                    # A list to keep track of all the factors we will keep for the next step
                    updatedFactorsList = list()            
        
                    for f in emissions:
                        # and select the ones that have the variable to be eliminated
                        if var in f.domain:
                            # join the factor `f` with the accumulator `newFactor`
                            newFactor = newFactor*f
                                
                        else:
                            # since the factor `f` doesn't contain `var`, we will keep it for next iteration
                            updatedFactorsList.append(f)
             
                    
                    newFactor = newFactor.marginalize(var) 
                    # append the new combined factor to the factor list
                    updatedFactorsList.append(newFactor)
                    # replace factorList with the updated factor list, ready for the next iteration
                    emissions = updatedFactorsList
                    
        # get state vars (to be marginalized later)

        state_vars = transition_factor.domain
        state_vars = list(state_vars)
        state_vars.remove(list(self.remap.keys())[0])
        state_vars = tuple(state_vars)

        # Setting evidence for transition - new in this implementation as transition includes door sensors
        transition_evi = transition_factor.evidence(**emission_evi)
        
        #print('Before Transition', self.state)
        #print('Transition Probabilities', transition_evi)
        
        # join with transition factor
        f = self.state*transition_evi

        # marginalize out old state vars, leaving only new state vars
        for var in state_vars:
            f = f.marginalize(var)

        # remap variables to their original names
        f.domain = tuple(self.remap[var] for var in f.domain)
        self.state = f

        #print('After Transition', self.state)

        # set emission evidence
        
        emissionFactor = Factor(tuple(), emissions[0].outcomeSpace)
        
        for factor in emissions:
            emissionFactor = emissionFactor*factor.evidence(**emission_evi) 
        
        #print('Emission Factor', emissionFactor)    
        
        # join with state factor
        f = self.state*emissionFactor

        print(f)

        # marginalize out emission vars
        for var in f.domain:
            if var not in state_vars:
                f = f.marginalize(var)
        self.state = f

        # normalize state - was commented in original code but want to use it to get prob dist at each step
        #print('Current Probabilities:')
        
        self.state = self.state.normalize()
        
        #print(self.state)

        index = self.state.outcomeSpace[class_var].index(value)
        
        # Extract the value probability at that step
        
        value_prob = self.state.table[index]

        return value_prob

    def forward_extra(self, missing = [], value = '0', class_var = 'r1', **emission_evi):
   
        emissions = list(self.emission.values())
        transition_factor = self.transition
        transition_extras = self.transition_extra
    
        if len(missing) > 0: #If there is at least 1 variable that needs to be marginalised
        
            for var in missing:
                if var in self.transition.domain: # Variable not in emission, but in other tables (door_sensor in this case)
                    transition_factor = transition_factor.marginalize(var)
                    for i, factor in enumerate(transition_extras):
                        if var in factor.domain:
                            transition_extras[i] = transition_extras.marginalize(var)
                            
                            
                            
                
                else:
                    # We create an empty factor as an accumulator
                    newFactor = Factor(tuple(), self.outcomeSpace)
                    # A list to keep track of all the factors we will keep for the next step
                    updatedFactorsList = list()            
        
                    for f in emissions:
                        # and select the ones that have the variable to be eliminated
                        if var in f.domain:
                            # join the factor `f` with the accumulator `newFactor`
                            newFactor = newFactor*f
                                
                        else:
                            # since the factor `f` doesn't contain `var`, we will keep it for next iteration
                            updatedFactorsList.append(f)
             
                    print(var)
                    print(newFactor.domain)
                    newFactor = newFactor.marginalize(var) 
                    # append the new combined factor to the factor list
                    updatedFactorsList.append(newFactor)
                    # replace factorList with the updated factor list, ready for the next iteration
                    emissions = updatedFactorsList
                    
        # get state vars (to be marginalized later)

        state_vars = transition_factor.domain
        state_vars = list(state_vars)
        state_vars.remove(list(self.remap.keys())[0])
        state_vars = tuple(state_vars)

        # Setting evidence for transition - new in this implementation as transition includes door sensors
        transition_evi = transition_factor.evidence(**emission_evi)
        
        print('Before Transition', self.state)
        print('Transition Probabilities', transition_evi)
        
        # join with transition factor
        f = self.state*transition_evi
        
        # Have to also join with P(d_{t+1},x) for forward pass
        for factor in transition_extras:
            factor = factor.evidence(**emission_evi)
            f = f*factor

        # marginalize out old state vars, leaving only new state vars
        for var in state_vars:
            f = f.marginalize(var)

        # remap variables to their original names
        f.domain = tuple(self.remap[var] for var in f.domain)
        self.state = f

        print('After Transition', self.state)

        # set emission evidence
        
        emissionFactor = Factor(tuple(), emissions[0].outcomeSpace)
        
        for factor in emissions:
            emissionFactor = emissionFactor*factor.evidence(**emission_evi) 
        
        print('Emission Factor', emissionFactor)    
        
        # join with state factor
        f = self.state*emissionFactor

        print(f)

        # marginalize out emission vars
        for var in f.domain:
            if var not in state_vars:
                f = f.marginalize(var)
        self.state = f

        # normalize state - was commented in original code but want to use it to get prob dist at each step
        print('Current Probabilities:')
        
        self.state = self.state.normalize()
        
        print(self.state)

        index = self.state.outcomeSpace[class_var].index(value)
        
        # Extract the value probability at that step
        
        value_prob = self.state.table[index]

        #time.sleep(2)

        return value_prob

    def forwardBatch(self, n, **emission_evi):
        '''
        emission_evi: A dictionary of lists, each list containing the evidence list for a variable. 
                         Use `None` if no evidence for that timestep
        '''
        history = []
        for i in range(n):
            # select evidence for this timestep
            evi_dict = dict([(key, value[i]) for key, value in emission_evi.items() if value[i] is not None])
            
            # take a step forward
            state = self.forward(**evi_dict)
            history.append(state)
        return history

    def viterbi(self, **emission_evi):
        '''
        This function is very similar to the forward algorithm. 
        For simplicity, we will assume that there is only one state variable, and one emission variable.
        '''

        # confirm that state and emission each have 1 variable 
        assert len(self.state.domain) == 1
        assert len(self.emission.domain) == 2
        assert len(self.transition.domain) == 2

        # get state and evidence var names (to be marginalized and maximised out later)
        state_var_name = self.state.domain[0]
        emission_vars = [v for v in self.emission.domain if v not in self.state.domain]
        emission_var_name = emission_vars[0]

        # join with transition factor
        f = self.state*self.transition

        # maximize out old state vars, leaving only new state vars
        f, prev = f.maximize(state_var_name, return_prev = True) 
        self.prev_history.append(prev) # save prev for use in traceback

        # remap variables to their original names
        f.domain = tuple(self.remap[var] for var in f.domain)
        self.state = f

        # set emission evidence
        emissionFactor = self.emission.evidence(**emission_evi)

        # join with state factor
        f = self.state*emissionFactor

        # marginalize out emission vars
        if emission_var_name in f.domain:
            f = f.marginalize(emission_var_name)
        self.state = f

        # normalize state (keep commented out for now)
        # self.state = self.state.normalize()

        self.history.append(self.state)

        return self.state

    def viterbiBatch(self, n,  **emission_evi):
        '''
        emission_evi: A dictionary of lists, each list containing the evidence list for a variable. 
                         Use `None` if no evidence for that timestep
        '''
        for i in range(n):
            # get evidence for this timestep
            evi_dict = dict([(key, value[i]) for key, value in emission_evi.items() if value[i] is not None])
            self.viterbi(**evi_dict)
        return self.history