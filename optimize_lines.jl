using JuMP, Ipopt
using HDF5
using Random

using HSL
import HSL_jll
bool = LIBHSL_isfunctional()
@info "HSL solvers are working: "*string(bool)

ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
println()
@warn("point to a Python env with alpha-beta-crown and auto-lirpa installed")
println()
ENV["JULIA_PYTHONCALL_EXE"] = "/home/schev/anaconda3/envs/alpha-beta-crown/bin/python" 
using PythonCall
using MathOptAI

sys = pyimport("sys")
println("Python version: ", sys.version)
println("Virtual Env Location: ", sys.prefix)

nn_file = "./24_bus_128node.pt"

Random.seed!(1)
M1 = randn(38,38)
Q  = M1'*M1

fid = h5open("./data.h5", "r")
x_lb = read(fid, "xnlb") .- 1e-4
x_ub = read(fid, "xnub")
close(fid)

include("functions.jl")

relaxation_methods = ["CROWN", "CROWN-IBP", "alpha-CROWN", "Forward+Backward"]
# %%
bound_method = relaxation_methods[3]
y_lb, y_ub, A_lb, b_lb, A_ub, b_ub = prepare_relaxed_model(x_lb, x_ub, nn_file, bound_method)

# ===================== y = NN(x)
local_obj    = solve_local_model(x_lb, x_ub, nn_file, Q)
relax_obj    = box_relaxation(y_lb, y_ub, Q)
lp_relax_obj = lp_relaxation(y_lb, y_ub, A_lb, b_lb, A_ub, b_ub, Q)

println()
println("\u2705 local: ", round(local_obj, sigdigits=6),".   \u2705 lp-relaxed: ", round(lp_relax_obj, sigdigits=6), ".   \u2705 box relaxed: ", round(relax_obj, sigdigits=6),)
