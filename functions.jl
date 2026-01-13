function solve_local_model(x_lb, x_ub, nn_file, Q)
    local_model = Model(Ipopt.Optimizer)
    set_attribute(local_model, "hsllib", HSL_jll.libhsl_path)
    set_attribute(local_model, "linear_solver", "ma27")
    set_optimizer_attribute(local_model, "tol",                 1e-3)
    set_optimizer_attribute(local_model, "acceptable_tol",      1e-3)

    @variable(local_model,            x[1:87])
    @constraint(local_model, x_lb .<= x[1:87] .<= x_ub)
    set_start_value.(x, 1*randn(87))

    predictor = MathOptAI.PytorchModel(joinpath(pwd(), nn_file))
    y, _ = MathOptAI.add_predictor(local_model, predictor, x; hessian=true, gray_box = true)

    @objective(local_model, Min, y'*Q*y)

    optimize!(local_model)
    println(objective_value(local_model))

    return objective_value(local_model)
end

function prepare_relaxed_model(x_lb, x_ub, nn_file, bound_method::String)
    numpy       = pyimport("numpy")
    torch       = pyimport("torch")
    autolirpa   = pyimport("auto_LiRPA")
    collections = pyimport("collections")

    BoundedModule      = autolirpa.BoundedModule
    BoundedTensor      = autolirpa.BoundedTensor
    PerturbationLpNorm = autolirpa.PerturbationLpNorm
    defaultdict        = collections.defaultdict

    nn_model = torch.load(nn_file, weights_only=false)
    nn_model.eval()

    x_lb_torch  = torch.from_numpy(numpy.array(x_lb)).float().unsqueeze(0)
    x_ub_torch  = torch.from_numpy(numpy.array(x_ub)).float().unsqueeze(0)
    ptb = PerturbationLpNorm(x_L=x_lb_torch, x_U=x_ub_torch)

    # this is the input around which we relax
    nn_input           = torch.from_numpy(numpy.array((x_lb_torch+x_ub_torch)/2)).float()
    bounded_model      = BoundedModule(nn_model, nn_input)
    nn_input_tensor    = BoundedTensor(nn_input, ptb)

    required_A = defaultdict(pybuiltins.set)
    required_A[bounded_model.output_name[0]].add(bounded_model.input_name[0])

    y_lb_autolirpa, y_ub_autolirpa, A = bounded_model.compute_bounds(x=(nn_input_tensor,), method=bound_method, return_A=true, needed_A_dict=required_A)

    y_lb = pyconvert(Array,(y_lb_autolirpa[0].detach().numpy()))
    y_ub = pyconvert(Array,(y_ub_autolirpa[0].detach().numpy()))

    # peel out the affine model
    Ab = A[bounded_model.output_name[0]][bounded_model.input_name[0]]

    A_lb = pyconvert(Array,Ab["lA"][0])
    b_lb = pyconvert(Array,Ab["lbias"][0])
    A_ub = pyconvert(Array,Ab["uA"][0])
    b_ub = pyconvert(Array,Ab["ubias"][0])

    return y_lb, y_ub, A_lb, b_lb, A_ub, b_ub
end

function box_relaxation(y_lb, y_ub, Q)
    relaxed_model = Model(Ipopt.Optimizer)
    @variable(relaxed_model, y[1:38])
    @constraint(relaxed_model, y_lb .<= y .<= y_ub)

    @objective(relaxed_model, Min, y'*Q*y)
    optimize!(relaxed_model)
    println(objective_value(relaxed_model))

    return objective_value(relaxed_model)
end

function lp_relaxation(y_lb, y_ub, A_lb, b_lb, A_ub, b_ub, Q)
    lp_relaxed_model = Model(Ipopt.Optimizer)
    @variable(lp_relaxed_model,   x[1:87])
    @constraint(lp_relaxed_model, x_lb .<= x[1:87] .<= x_ub)

    @variable(lp_relaxed_model, y[1:38])
    @constraint(lp_relaxed_model, y_lb .<= y .<= y_ub)

    @constraint(lp_relaxed_model,                   y .<= A_ub*x + b_ub)
    @constraint(lp_relaxed_model, A_lb*x + b_lb .<= y)

    @objective(lp_relaxed_model, Min, y'*Q*y)
    optimize!(lp_relaxed_model)
    println(objective_value(lp_relaxed_model))

    return objective_value(lp_relaxed_model)
end