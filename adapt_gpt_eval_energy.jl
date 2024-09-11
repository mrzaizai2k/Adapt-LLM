import ADAPT
import CSV
import DataFrames
import DataFrames: groupby
import Serialization
import LinearAlgebra: norm
import Graphs
import JSON
import JuMP, MQLib
using ProgressBars
import SimpleWeightedGraphs

if length(ARGS) < 3
    println("Usage: julia adapt_gpt_eval_energy.jl <input_fpath> <output_fpath> <n_nodes>")
    exit(1)
end

input_fpath = ARGS[1]
output_fpath = ARGS[2]
n_nodes = parse(Int, ARGS[3])

adapt_gpt_out_list = JSON.Parser.parsefile(
    input_fpath
);

pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_nodes);

function edgelist_to_graph(edgelist; num_vertices=0)
    if num_vertices == 0
        num_vertices = maximum(max(src, dst) for (src, dst, w) in edgelist)
    end
    g = SimpleWeightedGraphs.SimpleWeightedGraph(num_vertices)

    if length(edgelist[1]) == 3
        # Add edges with weights to the graph
        for (src, dst, w) in edgelist
            Graphs.add_edge!(g, src, dst, w)
        end
    else
        for (src, dst) in edgelist 
            w = 1
            Graphs.add_edge!(g, src, dst, w)
        end
    end
    
    return g
end

function graph_to_edgelist(g)
    weighted_edge_list = [
        (
            SimpleWeightedGraphs.src(e),
            SimpleWeightedGraphs.dst(e),
            SimpleWeightedGraphs.weight(e)
        ) for e in SimpleWeightedGraphs.edges(g)
    ]

    return weighted_edge_list
end 

iter = ProgressBar(1:length(adapt_gpt_out_list))

for graph_idx in iter

    #println(graph_idx)
    
    adapt_gpt_out_dict = adapt_gpt_out_list[graph_idx]

    edgelist = adapt_gpt_out_dict["graph_w_jl"];

    g = edgelist_to_graph(edgelist, num_vertices=n_nodes)

    e_list = graph_to_edgelist(g)

    H = ADAPT.Hamiltonians.maxcut_hamiltonian(n_nodes, e_list);
    
    ψ0 = ones(ComplexF64, 2^n_nodes) / sqrt(2^n_nodes); ψ0 /= norm(ψ0);

    adapt_gpt_energies_list = []
    
    for i in 1:(length(adapt_gpt_out_dict["q_circuits"]) + 1)
        if i <= length(adapt_gpt_out_dict["q_circuits"])
            #println(i)
            generated_list = adapt_gpt_out_dict["q_circuits"][i]
        else
            #println("ADAPT")
            generated_list =  adapt_gpt_out_dict["adapt_circuit"]
        end
        
        op_indices = []
        angle_values = []
        
        # Iterate over the list in steps of 4
        for j in 1:4:length(generated_list)
            push!(op_indices, generated_list[j+1])
            push!(angle_values, generated_list[j+3])
            push!(angle_values, generated_list[j+2])
        end
    
        angles = convert(Array{Float64}, angle_values);
    
        ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(0.01, H);
        #ansatz = ADAPT.Basics.Ansatz(1.0, pool) 
        
        for op_idx in op_indices
            push!(ansatz, pool[op_idx] => 0.0)
            # NOTE: this step adds both H and the pool operator to the ansatz
        end
    
        ADAPT.bind!(ansatz, angles);  #= <- this is your reconstructed ansatz =#
    
        # TEST: EVALUATE FINAL ENERGY - SHOULD MATCH LAST ENERGY FOR THAT "run"
        ψEND = ADAPT.evolve_state(ansatz, ψ0)
        E_final = ADAPT.evaluate(H, ψEND)
        #println("Iter $i energy = $E_final")

        #println(i)
        if i <= length(adapt_gpt_out_dict["q_circuits"])
            append!(adapt_gpt_energies_list, E_final)
        else
            #print("$i ADAPT")
            adapt_gpt_out_dict["ADAPT_energy_round"] = E_final
        end
    end

    adapt_gpt_out_dict["adapt_gpt_energies"] = adapt_gpt_energies_list
end

## Saving

adapt_gpt_out_list_json = JSON.json(adapt_gpt_out_list);

open(output_fpath,"w") do f 
    write(f, adapt_gpt_out_list_json) 
end