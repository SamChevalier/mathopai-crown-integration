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

nn_model = "./24_bus_128node.pt"

# %% =====================
local_model = Model(Ipopt.Optimizer)
set_attribute(local_model, "hsllib", HSL_jll.libhsl_path)
set_attribute(local_model, "linear_solver", "ma27")
set_optimizer_attribute(local_model, "tol",                 1e-3)
set_optimizer_attribute(local_model, "acceptable_tol",      1e-3)

fid = h5open("./data.h5", "r")
xn_lb = read(fid, "xnlb")
xn_ub = read(fid, "xnub")
close(fid)

@variable(local_model,   xn[1:87])
@constraint(local_model, xn_lb .<= xn[1:87] .<= xn_ub .+ 1e-3)

predictor = MathOptAI.PytorchModel(joinpath(pwd(), nn_model))
logit_zl, _ = MathOptAI.add_predictor(local_model, predictor, xn; hessian=true, gray_box = true)#, gray_box = true)#; reduced_space = true)#gray_box = true)#; gray_box=true) #; gray_box = true)#; reduced_space = true)#; gray_box = true)

Random.seed!(1)
M1 = randn(38,38)
Q = M1'*M1
@objective(local_model, Min, logit_zl'*Q*logit_zl)

optimize!(local_model)
println(objective_value(local_model))

# %% alternative: propagate bounds from x to logit_zl, then bound logit_zl
numpy     = pyimport("numpy")
torch     = pyimport("torch")
autolirpa = pyimport("auto_LiRPA")

BoundedModule      = autolirpa.BoundedModule
BoundedTensor      = autolirpa.BoundedTensor
PerturbationLpNorm = autolirpa.PerturbationLpNorm

model = torch.load(nn_model, weights_only=false)
model.eval()

lb  = torch.from_numpy(numpy.array(xn_lb)).float().unsqueeze(0)
ub  = torch.from_numpy(numpy.array(xn_ub)).float().unsqueeze(0)
ptb = PerturbationLpNorm(x_L=lb, x_U=ub)

# this is the input around which we relax
input         = torch.from_numpy(numpy.array((lb+ub)/2)).float().unsqueeze(0) 
bounded_model = BoundedModule(model, input)
input         = BoundedTensor(input, ptb)
lb_al, ub_al  = bounded_model.compute_bounds(x=(input,), method="alpha-CROWN")

# %% ======
relaxed_model = Model(Ipopt.Optimizer)
@variable(relaxed_model, logit_zl_bounded[1:38])
@constraint(relaxed_model, pyconvert(Array,(lb_al[0][0])) .<= logit_zl_bounded .<= pyconvert(Array,(ub_al[0][0])))

@objective(relaxed_model, Min, logit_zl_bounded'*Q*logit_zl_bounded)
optimize!(relaxed_model)
println(objective_value(relaxed_model))

# %% sanity check: do logit_zl values fall in the Auto-LIRPA bounds? Are the bounds valid?
println()
println(Bool(all(pyconvert(Array,(lb_al[0][0])) .<= value.(logit_zl) .<= pyconvert(Array,(ub_al[0][0])))))
println(Bool(objective_value(relaxed_model) < objective_value(local_model)))