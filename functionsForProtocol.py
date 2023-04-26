import random
import os
from qiskit import *
import numpy as np
import importlib
import copy
import math
from qiskit_aer.noise import NoiseModel #from qiskit.providers.aer.noise import NoiseModel
from qiskit_aer import AerSimulator #from qiskit.providers.aer import AerSimulator
from itertools import *
from qiskit.test.mock import *
import warnings
import braket.circuits
from datetime import datetime
import boto3
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import Circuit
import re #regex for parsing circuit to ionq
import collections
from functools import reduce

# HELPER FUNCTIONS
"""
Normalize count results in weight dictionaries to sum to 1 and thus correspond to probabilities of measurement.
Can be sorted states from largest to smallest probability. Can round results to 'digits' decimals.
"""
def outputformat(d, target=1.0, normalizeOutput=True, sortOutput=True, digits=None):
    #round results to number of 'digits' after comma, do not round if digits=None
   d = {k: v for k, v in d.items() if v != 0} #take out elements with value==0
   raw = sum(d.values())
   factor = target/raw
   if normalizeOutput and sortOutput:
       if digits == None:
            return {key: (value * factor) for key, value in sorted(d.items(), reverse=True, key=lambda item: item[1])}
       else:
            return {key:round(value*factor,digits) for key, value in sorted(d.items(), reverse=True, key=lambda item: item[1])}
   elif normalizeOutput and not sortOutput:
       if digits == None:
            return {key: (value * factor) for key, value in d.items()}
       else:
            return {key:round(value*factor,digits) for key, value in d.items()}
   elif not normalizeOutput and sortOutput:
       return {key: value for key, value in
               sorted(d.items(), reverse=True, key=lambda item: item[1])}
   else:
       return d

"""
Convert a dictionary with state measurement probabilities to a list. Index 0 corresponds to key '000', index 1 to '001' etc. 
Always has length 2^N, even if not all states were measured in dict.
"""
def probdict2list(dict):
    N = len(list(dict.keys())[0]) #get first key in the dict -> get number of characters in its string = N
    problist = list(np.repeat(0, 2**N))
    for key in dict.keys():
        keyint = int(key, 2) #already 0-indexed and little endian, i.e. 000->0, 001->1,...,111->7
        problist[keyint] = dict[key]
    return problist

"""
Parser that read BoolNet formatted txt file with rules and generates a new .py file including all the update functions as required in tweedledum
"""
def parse_BoolNetRules(rulestxt, saveDir=None):

    #Parsing of txt file, needs to have exactly one line break after last rule
    pathsplit_dot = str.rsplit(rulestxt, sep=".")
    splitfilename = pathsplit_dot[len(pathsplit_dot)-2]
    pathsplit_slash = str.rsplit(splitfilename, sep="/")
    netname = pathsplit_slash[len(pathsplit_slash) - 1]

    #####
    with open(rulestxt, 'a') as f:
        f.write(f'\n')
    tempoutputtxt = "tempoutput.txt"
    # Remove superfluous empty lines, write to new file, only one single linebreak after last rule
    with open(rulestxt, 'r') as r, open(tempoutputtxt, 'w') as o:
        for line in r:
            if not line.isspace():
                o.write(line)
    # Rename output file to name of input file
    os.replace(tempoutputtxt, rulestxt)
    ######

    rulesvar = open(rulestxt, 'r')
    lines = rulesvar.readlines()
    count = 0

    genenamestring = [0 for i in range(len(lines))]
    genecount = 0
    for line in lines:
        splitline = line.split(sep=",")
        genenamestring[genecount] = splitline[0].lower() #must have lowercase gene names
        genecount += 1
    genenamestring = genenamestring[1:]

    genenamestring_intmap = ""
    # genenamestring_intmap = A: Int1, B: Int1, C: Int1
    for g in genenamestring:
        genenamestring_intmap += g
        genenamestring_intmap += ": Int1, "
    #Final two chars, comma + space not needed at end
    genenamestring_intmap = genenamestring_intmap[:-2]

    # Generate new python file
    if saveDir == None:
        newpyfilename = netname + "_rulefile.py"
    else:
        newpyfilename = saveDir + netname + "_rulefile.py"

    with open(newpyfilename, 'w+') as pyfileRules:
        pyfileRules.write("#Libraries: \n")
        pyfileRules.write("from qiskit import * \n")
        pyfileRules.write("from qiskit.circuit import classical_function, Int1 \n")
        pyfileRules.write(" \n")
        pyfileRules.write("#Regulatory functions to synthesize into circuits using Qiskit: \n")
        for line in lines: #read every rule in txt file
            if count == 0:
                count += 1
                continue #skip targets, factors header line
            else:
                pyfileRules.write("@classical_function \n")
                pyfileRules.write("def g" + str(count-1) + "_update(" + genenamestring_intmap + ") -> Int1:\n")

                BN_regstring = line.split(sep=",")[1]
                BN_regstring = BN_regstring[1:-1] #remove first space and final linebreak from rulestring

                #replace & | and ! symbols
                BN_regstring = BN_regstring.replace("&", "and")
                BN_regstring = BN_regstring.replace("|", "or")
                BN_regstring = BN_regstring.replace("!", "not ")

                pyfileRules.write("\t return (" + BN_regstring.lower() + ") \n")
                pyfileRules.write(" \n")

                count += 1

    pyfileRules.close()
    #load all single gene update functions from parsed py file
    with open(newpyfilename) as f:
        exec(compile(f.read(), newpyfilename, "exec"))




#Function for notebooks: simulate_statetransitions
"""
Generate circuit that inits starting states with Ry gates, to be composed with single/multistep circuits + .measure() statement
"""
def initActivityCircuitGenerator(activities, Tmax=1, thetatype="angle", method="exact"):
    # map starting state gene activities=[0,1] (KO->unperturbed=0.5->OE) to theta=[0,pi]
    # use activities=np.repeat(0.5,n) for allH default init
    n = len(activities)
    initcirc = QuantumCircuit((Tmax+1) * n, n)
    if thetatype == "angle":
        #Map [0,180] to [0,1]
        activities = [activity/180 for activity in activities]
    elif thetatype == "radian":
        #Map [0,pi] to [0,1]
        activities = [activity/np.pi for activity in activities]
    else:
        raise ValueError("Please choose 'angle' or 'radian' for thetatype.")

    for g in range(len(activities)):
            initcirc.ry(theta=activities[g]*np.pi, qubit=g)
    return(initcirc)

"""
Given a text file with update rules in BoolNet-format, generates a 2*n,n quantum circuit that updates the entire network by one timestep if executed.
If update="asynchronous" a random update order is generated and circuits use the updated state of previously updated nodes.
"""
def synthesizeFullNetworkUpdateCircuit(rulestxt, update="synchronous", includeClassicalRegister=True, updateorder=None):
    parse_BoolNetRules(rulestxt)

    #g0_update() etc functions only become available inside this function when exec is called again explicitely
    # Fixed to work for arbitrary paths to subfolders
    pathsplit_dot = str.rsplit(rulestxt, sep=".")
    splitfilename = pathsplit_dot[len(pathsplit_dot) - 2]
    pathsplit_slash = str.rsplit(splitfilename, sep="/")
    netname = pathsplit_slash[len(pathsplit_slash) - 1]

    newpyfilename = netname + "_rulefile"
    importrules = __import__(newpyfilename)

    rulesvar = open(rulestxt, 'r')
    lines = rulesvar.readlines()
    n = len(lines)-1 #-1 to account for header line

    #If circuit has to be turned into a gate then it must not include a classical register
    if includeClassicalRegister:
        FullCircuit = QuantumCircuit(2 * n, n)
    else:
        FullCircuit = QuantumCircuit(2 * n)
    qr = QuantumRegister(2*n)
    cr = ClassicalRegister(n)

    CircuitList = list(np.repeat(0, n))

    if update == "synchronous":
        for g in range(n):
            exec("CircuitList[" + str(g) + "] = importrules.g" + str(g) + "_update.synth()") # individual gene's update circuit with n+1 qubits (n inputs, one output)
            if includeClassicalRegister:
                CircuitList[g].add_register(cr)
            # All individual circuit QuantumCircuit objects are synthesized, stored in list, and have cr added
            # FullCircuit = FullCircuit.compose(GenegUpdate, [0,..., n-1, n+g], [0,...,n-1]) #n+g = n,..., 2n-1
            outputqubitlist = list(range(n))
            outputqubitlist.append(n+g)
            if includeClassicalRegister:
                FullCircuit = FullCircuit.compose(CircuitList[g], outputqubitlist, list(range(n)))
            else:
                FullCircuit = FullCircuit.compose(CircuitList[g], outputqubitlist)
            #second argument=qubits of self to compose onto, last qubit is varying output of gX(t+1), third argument is classical register (unchanging)

        return (FullCircuit)

    elif update == "asynchronous":
        if updateorder == None: #if no order is specified, generate a random order
            updateorder = list(range(n))
            random.shuffle(updateorder)
        elif len(updateorder) != n:
            print("updateorder needs to be of length n")
        print("Update order of the asynchronous circuit is " + str(updateorder))
        outputqubitlist = list(range(n+1))

        for g in range(n):
            gene2update = updateorder[g]
            exec("CircuitList[" + str(gene2update) + "] = importrules.g"+str(gene2update)+"_update.synth()") # individual gene's update circuit with n+1 qubits (n inputs, one output)
            CircuitList[gene2update].add_register(cr)
            # All individual circuit QuantumCircuit objects are synthesized, stored in list, and have cr added
            outputqubitlist[n] = n+gene2update
            FullCircuit = FullCircuit.compose(CircuitList[gene2update], outputqubitlist, list(range(n)))
            outputqubitlist[gene2update] = n+gene2update
            #second argument=qubits of self to compose onto, last qubit is varying output of gX(t+1), third argument is classical register (unchanging)

        return (FullCircuit)

    else:
        return ("Not a valid updating scheme")


#SYNCHRONOUS STATE TRANSITIONS
#TODO: Renamed from "allparam_exact_multiTransition_synchronous" to "quantumStateTransition"
"""
Exact multistep circuit transitions which can distinguish if Ry gates should serve only as initial expression bias or be used for true perturbations.
Also allows for addition of noise models. Performs synchronous state transitions.
"""
def quantumStateTransition(rulestxt, Tmax=2, nrshots=100, initActivities=None, initPerturbed=None,
                                   thetatype="linear", normalizeOutput=True, sortOutput=True, backend=None, optimization_level=0,
                                   approximation_degree=None, returnCircuitDepth=False, seed_transpiler=None, seed_simulator=None,
                                   returnCircuitOnly=False):
    if isinstance(rulestxt, QuantumCircuit):
        circuit = rulestxt #Directly provided circuit
    else:
        circuit = synthesizeFullNetworkUpdateCircuit(rulestxt, update="synchronous") #Provided path to rules -> synthesize circuit
    #First generate the MultiStepCircuit:
    n = int(circuit.num_qubits/2)
    totalqubits = (Tmax + 1) * n #Needed 2*n qubits for single transition, now (T+1)*n qubits needed for full circuit
    cr = ClassicalRegister(n)

    if initPerturbed == None:
        initPerturbed = list(np.repeat(False, n))

    if initActivities == None:
        InitCircuit = QuantumCircuit(totalqubits, n)
        #default init all-H
        for q in range(0,n):
            InitCircuit.h(q)
    else:
        InitCircuit = initActivityCircuitGenerator(activities=initActivities, Tmax=Tmax, thetatype=thetatype) #Returns only 2n circuit, not (T+1)*n, need to add (T-1)*n qubits

        #all non 0.5 init genes now had an Ry gate added, no need to expand initActivityCircuitGenerator
        #for non-perturbed genes, keep calculating transitions, for perturbed ones, keep referring to these original states

    #initPerturbed should be a boolean vector of length n
    # If True, gene g will not be updated (perturbation), if False, gene g will be updated starting from biased Ry state normally

    if Tmax==0:
        InitCircuit.measure(list(range(0, n)), list(range(0, n)))
        Aer.backends()
        result = execute(InitCircuit, backend=Aer.get_backend('qasm_simulator'),
                         shots=nrshots, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()  # execute(circuit, simulator, shots)
        weightDict = result.get_counts(InitCircuit)
        return (outputformat(weightDict, normalizeOutput=normalizeOutput, sortOutput=sortOutput, digits=None))


    indices2compose = list(range(0, 2*n)) #list [0,...,2n-1], correct
    MultiStepCircuit = InitCircuit.compose(circuit, indices2compose, list(range(0, n))) #InitCircuit+first transition step done, t+1 outputs are at qn to q2n-1

    #Synchronous circuits, construct once, measure nrshots times
    for t in range(2, Tmax+1): #transitions nr 2-Tmax
        indices2compose = list(range((t-1)*n,(t+1)*n)) #(t-1)*n to (t+1)*n-1, 2n qubits in total to compose
        for g in range(n): #only need to update first n out of list of length 2n
            if initPerturbed[g] == False:
                #unperturbed case
                pass
            else:
                #perturbed case
                #print("Overwriting index g=" + str(g) + " (0-indexed)")
                indices2compose[g] = g

        #MultiStepCircuit = MultiStepCircuit.compose(circuit, indices2compose, cr) #OLD Call here
        MultiStepCircuit = MultiStepCircuit.compose(circuit, qubits=indices2compose, clbits=list(range(n))) #NEW Changed clbits

    #Changed list of qubits to measure, so no reset+overwrite is needed
    qubits2measure = list(range(Tmax*n, (Tmax+1)*n)) #length n
    for g in list(range(n)):
        if initPerturbed[g] == True:
            qubits2measure[g] = g #measure initial superposition instead of final output that was not yet overwritten after last transition

    MultiStepCircuit.measure(qubits2measure, list(range(n)))

    #return only the circuit object instead of the dictionary with measurement results for use in Runtime
    if returnCircuitOnly:
        return(MultiStepCircuit)
    else:
        Aer.backends()
        if backend != None:
            print("Transpiling circuit to backend " + str(backend))
            MultiStepCircuit = transpile(MultiStepCircuit, backend=backend, optimization_level=optimization_level,
                                         approximation_degree=approximation_degree, seed_transpiler=seed_transpiler)
            #Adding noise if wanted
            print("Adding noise of backend " + str(backend))
            sim_FakeBackend = AerSimulator.from_backend(backend)
            result = sim_FakeBackend.run(MultiStepCircuit, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()
        else:
            result = execute(MultiStepCircuit, backend=Aer.get_backend('qasm_simulator'),
                             shots=nrshots, seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()

        countDict = result.get_counts(MultiStepCircuit)
        countDict = outputformat(countDict, normalizeOutput=normalizeOutput, sortOutput=sortOutput, digits=None)

        if returnCircuitDepth == True:
            print("The circuit has a depth of " + str(MultiStepCircuit.depth()))

        return(countDict)


"""
Helper function for Grover state transition inversion. 
Returns transition circuit for multiple transition steps without Hadamard initialization or measurement so that circuit can be used as a gate.
"""
def generate_exact_multiTransitionGate(rulestxt, Tmax=1):
    #Get multistep circuit without init, measurement or classical register from new functions so that it can be turned into a gate
    circ = synthesizeFullNetworkUpdateCircuit(rulestxt, update="synchronous", includeClassicalRegister=False)
    n = circ.num_qubits//2 #number nodes in network
    finalcirc = QuantumCircuit(n*(Tmax+1))
    for step in range(1,(Tmax+1)):
        finalcirc = finalcirc.compose(circ, list(range(n*(step-1), n*(step+1))))
    return finalcirc


"""
Function which implements G iterations of a Grover Oracle and Diffuser operator.
The marked state (i.e. an attractor) is the solution element. Grover search will then amplify probabilities of states which lie nrTransitions step before the marked state in the STG.
That is, it effectly performs nrTransitions 'inverse' state transitions.
The optimal value for G will depend on the (unknown) number of predecessor states.
"""
def generate_groverSTGinversion_circuit(rulestxt, nrTransitions=1, markedState=None, G=1):
    #Transition circuits as gates to be used as blocks in Grover circuit
    transitionCircuit = generate_exact_multiTransitionGate(rulestxt, Tmax=nrTransitions)
    transitiongate = transitionCircuit.to_gate()
    transitiongate_inv = transitiongate.inverse()

    n = transitionCircuit.num_qubits//(nrTransitions+1)
    GroverCircuit = QuantumCircuit(transitionCircuit.num_qubits + 1, n)
    minusqubitindex = transitionCircuit.num_qubits

    # Put last qubit in |minus> state via X->H
    GroverCircuit.x(minusqubitindex)
    GroverCircuit.h(minusqubitindex)

    # Init genes with H layer
    for q in range(n):
        GroverCircuit.h(q)

    markedState = markedState[::-1]
    for GroverIteration in range(G):
        ### ORACLE
        # Transition gate block
        GroverCircuit.append(transitiongate, list(range(n*(nrTransitions+1))))

        # Loop for X gates corresponding to markedState on output register
        outputregister = list(range(n*nrTransitions, n*(nrTransitions+1)))

        for g in range(n):
            #need X gate if there is a 0 in markedState, no X gate if there is a 1
            if markedState[g] == 0:
                GroverCircuit.x(outputregister[g])

        # MCX gate over output of transition circuit, target |minus> ancilla
        GroverCircuit.mcx(outputregister, minusqubitindex)

        # Loop for X gates corresponding to markedState
        for g in range(n):
            #need X gate if there is a 0 in markedState, no X gate if there is a 1
            if markedState[g] == 0:
                GroverCircuit.x(outputregister[g])

        # Inverse Transition gate block
        GroverCircuit.append(transitiongate_inv, list(range(n * (nrTransitions + 1))))

        ###DIFFUSER
        # Hadamard layer for initital genes
        for q in range(n):
            GroverCircuit.h(q)

        # X layer for initial genes
        for q in range(n):
            GroverCircuit.x(q)

        # MCX gates over initial genes, target |minus> ancilla
        GroverCircuit.mcx(list(range(n)), minusqubitindex)

        # X layer for initial genes
        for q in range(n):
            GroverCircuit.x(q)

        # Hadamard layer for initial genes
        for q in range(n):
            GroverCircuit.h(q)


    # Add measurement operators for initial gene qubits
    GroverCircuit.measure(list(range(n)), list(range(n)))

    return(GroverCircuit)


"""
Function to translate measured bitstrings from Quantum Counting circuit into probabilities for M as the number of predecessor states with a t qubit readout register.
"""
def calculateMfromBitstring(outputDict, t, n, verbose=True):
    measured_int_dict = {int(k,2):float(v) for k,v in outputDict.items()} #dict with keys = integers, values = floats/probabilities
    # int(k,2) counts e.g. 01101 as 13 -> q0 i.e. least valued bit is written at the right
    N = 2**n
    theta_dict = {(k*2*np.pi/(2**t)):float(v) for k,v in measured_int_dict.items()} #dict with keys = phase angles
    NminusM_dict = {(N*np.sin(k/2)*np.sin(k/2)):float(v) for k,v in theta_dict.items()} #dict with keys = N-M values
    M_dict = {(N-k):float(v) for k,v in NminusM_dict.items()}
    #M_dict Returns non-solutions instead of solutions since implemented diffuser -U_s instead of U_s -> Need to calculate M = N - returnedValue
    solutions_dict = outputformat({round(k):float(v) for k,v in M_dict.items()}) #returns solutions M, rounded to integers to aggregate some keys and their weights together

    weightedAverage = 0
    for k in solutions_dict.keys():
        weightedAverage += k*solutions_dict[k]

    solutions_array = [0] * (N + 1) #0-N are all possible solutions
    for k in solutions_dict.keys():
        solutions_array[k] = solutions_dict[k]

    if verbose:
        print("The weighted average of the returned M dictionary is " + str(round(weightedAverage,3)))
        print("The standard deviation of the measured distribution is " + str(round(np.std(solutions_array),3)))
        print("The most likely measurement of the circuit corresponds to M = " + str(max(solutions_dict, key=solutions_dict.get)))
    return(solutions_dict)


"""
Function which generates the Grover operator G for a given number of inverted transitions and a marked state of interest.
Returns a circuit (without initialization gates) which can then be exponentiated and controlled for use in the Quantum Counting algorithm.
"""
def GroverIterationFromTransitionCircuit(transitionCircuit, markedState, nrTransitions):
    transitiongate = transitionCircuit.to_gate()
    transitiongate_inv = transitiongate.inverse()

    n = transitionCircuit.num_qubits // (nrTransitions + 1)
    GroverCircuit = QuantumCircuit(transitionCircuit.num_qubits + 1)
    minusqubitindex = transitionCircuit.num_qubits

    ### ORACLE
    # Transition gate block
    GroverCircuit.append(transitiongate, list(range(n * (nrTransitions + 1))))

    # Loop for X gates corresponding to markedState on output register
    outputregister = list(range(n * nrTransitions, n * (nrTransitions + 1)))
    markedState = markedState[::-1]

    for g in range(n):
        # need X gate if there is a 0 in markedState, no X gate if there is a 1
        if markedState[g] == 0:
            GroverCircuit.x(outputregister[g])

    # MCX gate over output of transition circuit, target |minus> ancilla
    GroverCircuit.mcx(outputregister, minusqubitindex)

    # Loop for X gates corresponding to markedState
    for g in range(n):
        # need X gate if there is a 0 in markedState, no X gate if there is a 1
        if markedState[g] == 0:
            GroverCircuit.x(outputregister[g])

    # Inverse Transition gate block
    GroverCircuit.append(transitiongate_inv, list(range(n * (nrTransitions + 1))))

    ###DIFFUSER
    # Hadamard layer for initital genes
    for q in range(n):
        GroverCircuit.h(q)

    # X layer for initial genes
    for q in range(n):
        GroverCircuit.x(q)

    # MCX gates over initial genes, target |minus> ancilla
    GroverCircuit.mcx(list(range(n)), minusqubitindex)

    # X layer for initial genes
    for q in range(n):
        GroverCircuit.x(q)

    # Hadamard layer for initial genes
    for q in range(n):
        GroverCircuit.h(q)

    return(GroverCircuit)


"""
Generate circuit for a quantum counting algorithm with a readout register of r_registerLen qubits. Performs nrTransitions inverted transitions from the markedState.
Returns a dictionary with probabilities for the resulting number of predecessor states M (rounded to integers).
Increasing the size of the readout register will yield more accurate results.
"""
def QuantumCountingAlgo(rulestxt, nrTransitions, markedState, r_registerLen, nrshots = 1000, verbose=True, seed_transpiler=None, seed_simulator=None):
    #Set default parameters:
    do_swaps_QFT = True
    approximation_degree_QFT = 0

    # Full QCounting circ generation U^2^(t-1), U^2^(t-2)...U^2^0 -> QFTdgr -> measure r register -> resulting state bitstring encodes |2^r ANGLE>
    transitionCircuit = generate_exact_multiTransitionGate(rulestxt, Tmax=nrTransitions)
    G = GroverIterationFromTransitionCircuit(transitionCircuit, markedState, nrTransitions)
    n = transitionCircuit.num_qubits // (nrTransitions+1)
    QCountingCirc = QuantumCircuit((nrTransitions+1)*n+1+r_registerLen, r_registerLen) #full circuit with measurement + Grover registers
    MeasurementRegister = list(range(r_registerLen))
    GroverRegister = list(range(r_registerLen, QCountingCirc.num_qubits))

    #Initialize MeasurementRegister with H gates as well
    for q in MeasurementRegister:
        QCountingCirc.h(q)
    #for q in GroverRegister:
    for q in GroverRegister[:n]: #Transition output register(s) of Grover should not be initialised with H -> input n genes need H layer but output should be mapped onto 0 qubits!!
        QCountingCirc.h(q)
    #for q in GroverRegister[:-1]: #No H gate for last qubit which is the minus state
    #    QCountingCirc.h(q)

    #Final qubit initialised as |minus> once outside of all cGrover iterators
    QCountingCirc.x(QCountingCirc.num_qubits-1)
    QCountingCirc.h(QCountingCirc.num_qubits - 1)

    #Compose onto circuit in a loop with t register as control qubits
    for t in range(r_registerLen):
        powerof2Gcircuit = QuantumCircuit(G.num_qubits)
        powerof2Gcircuit = powerof2Gcircuit.compose(G, list(range(G.num_qubits)))
             #for t=0 -> return this immediately
        for compose in range(2**t - 1):
            #for t=1 -> G^2^1 = G^2 compose 1 more time
            #for t=2 -> G^2^2 = G^4 compose 3 more times --> compose 2^t-1 more times G after G into
            powerof2Gcircuit = powerof2Gcircuit.compose(G, list(range(G.num_qubits)))

        powerof2Ggate = powerof2Gcircuit.to_gate()
        powerof2cGgate = powerof2Ggate.control(num_ctrl_qubits=1)
        cGroverRegister = [t] + GroverRegister #append to beginning of list, G^2^0 controlled by q0
        QCountingCirc = QCountingCirc.compose(powerof2cGgate, cGroverRegister)

    from qiskit.circuit.library import QFT
    QFTdgrCirc = QFT(num_qubits=r_registerLen, approximation_degree=approximation_degree_QFT, do_swaps=do_swaps_QFT, inverse=True)
    QCountingCirc = QCountingCirc.compose(QFTdgrCirc, list(range(r_registerLen)))
    QCountingCirc.measure(MeasurementRegister, MeasurementRegister)

    result = execute(QCountingCirc, backend=Aer.get_backend('qasm_simulator'), shots=nrshots,
                     seed_transpiler=seed_transpiler, seed_simulator=seed_simulator).result()
    measured_int_dict = result.get_counts(QCountingCirc)
    M_dict = calculateMfromBitstring(measured_int_dict, r_registerLen, n, verbose=verbose)
    return(M_dict)

"""
Classical fidelity to compare two probability distributions over the same index set, given as F_s by Lubinski et al.
"""
def classical_fidelity(p,q):
    dicts = [p, q]
    keysUnion = list(reduce(set.union, map(set, map(dict.keys, dicts))))
    #print(keysUnion)

    for k in keysUnion:
        if k not in list(p.keys()):
            #print("Key " + str(k) + " is not in the dict p!")
            p[k] = 0
        if k not in list(q.keys()):
            #print("Key " + str(k) + " is not in the dict q!")
            q[k] = 0

    #print("Matching dicts p and q:")
    ordered_p = dict(collections.OrderedDict(sorted(p.items())))
    ordered_q = dict(collections.OrderedDict(sorted(q.items())))

    #Calculate fidelity
    fidelity = 0
    for k in keysUnion:
        fidelity += np.sqrt(ordered_p[k]*ordered_q[k])
    return fidelity**2

"""
Normalized fidelity as defined by Lubinski et al. (2021).
"""
def normalized_fidelity(idealDistr, outputDistr):
    n = len(list(idealDistr.keys())[0])
    N = 2**n
    #generate uniform distribution dictionary for n-qubit system
    bitstringkeys = [''.join(i) for i in product('01', repeat=n)]
    uniformvalues = [1/N]*N
    uniformDistr = dict(zip(bitstringkeys, uniformvalues))
    return (classical_fidelity(idealDistr, outputDistr) - classical_fidelity(idealDistr, uniformDistr))/(1 - classical_fidelity(idealDistr, uniformDistr))


"""
Mapping of gate names between Qiskit and Amazon Braket/IonQ.
"""
#keys are qiskit gates -> vals are ionq names that they should be parsed to
#v gate in ionq is qiskits square root of not sx, vi is sxdg
gatenameMapping_Qiskit_IonQ = {"rzz":"zz", "sxdg":"vi", "rxx":"xx", "z":"z", "swap":"swap", "sx":"v",
                   "y":"y", "rx":"rx", "rz":"rz", "s":"s", "ry":"ry", "yy":"yy", "tdg":"ti",
                   "h":"h", "i":"i", "x":"x", "sdg":"si", "cx":"cnot", "t":"t"}
#ionq_basisgates_qiskitnames = ['rzz', 'sxdg', 'rxx', 'z', 'swap', 'sx', 'ry', 'rx', 'rz', 's', 'ry', 'yy', 'tdg', 'h', 'i', 'x', 'sdg', 'cx', 't']

"""
Takes a Qiskit circuit and transpiles it to the IonQ set of basis gates, given a mapping dictionary of gate names between these two frameworks.
The transpiled circuit's QASM representation is used to create a braket.circuits.Circuit() object with these exact gates, which can then be run on Braket.
"""
def QiskitQASM2IonQCompiler(QiskitCircuit, gatenameMapping):
    TranspiledQiskitCircuit = transpile(QiskitCircuit, basis_gates=list(gatenameMapping.keys()))
    IonQCircuit = braket.circuits.Circuit()
    QiskitQASM = TranspiledQiskitCircuit.qasm().split("\n") #list of operations
    #Parse its instruction into braket ie line cx q[4],q[7] -> cnot(4,7)
    totallines = len(QiskitQASM)
    for linenr in range(4,totallines-1):
        operatorstr = QiskitQASM[linenr]
        #print("operatorstr = " + operatorstr)
        braketstr = ""
        opname_qubits = operatorstr.split(" ")
        opname = opname_qubits[0]
        qubits = opname_qubits[1]
        #split operatorname on ( if it occurs
        # -> first is gate name, translate to ionq gate name,
        # second is angle -> replace pi with np.pi
        gatename_operand = opname.split("(")
        gatename = gatename_operand[0]
        operand = ""
        if len(gatename_operand) > 1:  # e.g. rz(-pi/8) q...; but not if e.g. h q...; or cx q...;
            operand = gatename_operand[1]
            operand = operand.replace("pi", "np.pi")
        qubitnumbers = re.findall(r'\d+', qubits)
        braketstr += gatenameMapping[gatename] + "("
        for q in qubitnumbers:
            braketstr += str(q)
            braketstr += ","
        braketstr += operand
        braketstr = braketstr[:-1]
        braketstr += ")"
        #print("braketstr = " + braketstr)
        exec("IonQCircuit." + braketstr)

    return IonQCircuit

"""
Function for parsing results from IonQ device back into same notation as Qiskit.
IonQ always measures all qubits, and its order/endian is opposite to Qiskit.
"""
def ionqDict2qiskitNotation(ionqDict, qubitindices, invertEndian=True, normalizeOutput=True, sortOutput=True):
    qiskitDict = {}
    for key in ionqDict.keys():
        reorderedShortKey = ""
        for qind in qubitindices:
            reorderedShortKey += key[qind]
        if reorderedShortKey not in list(qiskitDict.keys()):
            qiskitDict[reorderedShortKey] = 0
        qiskitDict[reorderedShortKey] += ionqDict[key]

    qiskitDict = outputformat(qiskitDict, normalizeOutput=normalizeOutput, sortOutput=sortOutput)

    if invertEndian == True: #go back to qiskit endian
        invertedQiskitDict = {}
        for key in list(qiskitDict.keys()):
            invertedkey = key[::-1]
            invertedQiskitDict[invertedkey] = qiskitDict[key]
        return invertedQiskitDict
    else:
        return qiskitDict

