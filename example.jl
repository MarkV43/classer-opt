using JuMP, Ipopt, LinearAlgebra

gamma = 1.0
xs = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]
ys = [[0.0, 1.0], [1.0, 1.0], [0.5, 1.5]]

model = Model(Ipopt.Optimizer)

@variable(model, a[1:2])
@variable(model, b, start = -1.0)
@variable(model, u[1:3] >= 0)
@variable(model, v[1:3] >= 0)

# give nonzero starts
set_start_value.(a, [-1.052e-11, -2.0])
set_start_value.(u, [1.4340678, 1.4340684, 2.1560354])
set_start_value.(v, [1.7993226, 1.7993227, 1.338834])

sum_uv = zero(AffExpr)

for ui in u
    add_to_expression!(sum_uv, ui);
end
for vi in v
    add_to_expression!(sum_uv, vi);
end

for (i,x) in enumerate(xs)
    @constraint(model, dot(a, x) - b >= 1 - u[i])
end
for (i,y) in enumerate(ys)
    @constraint(model, dot(a, y) - b <= -1 + v[i])
end

@objective(model, Min,
    sqrt(a[1]^2 + a[2]^2) + gamma*(sum_uv)
)

optimize!(model)

println("status = ", termination_status(model))
println("a = ", value.(a), " b = ", value(b))
println("u = ", value.(u), " v = ", value.(v))
println("obj = ", objective_value(model))

println("constraint = ", value(dot(a, xs[1]) - b), " >= ", value(1 - u[1]))
println("constraint = ", value(dot(a, xs[2]) - b), " >= ", value(1 - u[2]))
println("constraint = ", value(dot(a, xs[3]) - b), " >= ", value(1 - u[3]))
println("constraint = ", value(dot(a, ys[1]) - b), " <= ", value(-1 + v[1]))
println("constraint = ", value(dot(a, ys[2]) - b), " <= ", value(-1 + v[2]))
println("constraint = ", value(dot(a, ys[3]) - b), " <= ", value(-1 + v[3]))
