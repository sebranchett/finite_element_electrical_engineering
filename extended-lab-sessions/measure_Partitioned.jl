using PartitionedArrays
using Plots
using IterativeSolvers
using LinearAlgebra
using BenchmarkTools 

#..first generate the row partition
np = 3
N = 1000
ranks = LinearIndices((np,))
row_partition = uniform_partition(ranks,N+1)

#..construct the mesh: see before 
h = 1/N; 
x = Vector(0:h:1); 

#..Mesh with points and edges 
#..point holds the coordinates of the left and right node of the element
#..edges holds the global indices of the left and right node of the element
points = collect( [x[i], x[i+1]] for i in 1:length(x)-1) 
edges = collect( [i, i+1] for i in 1:length(x)-1); 

#..Set the source function 
fsource(x) = x*(x-1); 

function compute_rhs_vector()
    IV = map(row_partition) do row_indices
        I,V = Int[], Float64[]
        for global_row in local_to_global(row_indices)
            if global_row == 1
                xl, xr = points[global_row,:][1]
                floc = (xr-xl) * [fsource(xl) fsource(xr)]
                v = floc[1]
            elseif global_row == N+1
                xl, xr = points[global_row-1,:][1]
                floc = (xr-xl) * [fsource(xl) fsource(xr)]
                v = floc[2]
            else
                xll, xrl = points[global_row-1,:][1]
                flocl = (xrl-xll) * [fsource(xll) fsource(xrl)]
                xlr, xrr = points[global_row,:][1]
                flocr = (xrr-xlr) * [fsource(xlr) fsource(xrr)]
                v = flocr[1] + flocl[2]
            end
            push!(I,global_row)
            push!(V,v)
        end
        I,V
    end
    I,V = tuple_of_arrays(IV)
    f = pvector!(I,V,row_partition) |> fetch
    #..handle the boundary conditions in the right-hand side vector 
    # right-hand side vector already satisfied:
    # f[1]   = 0;     f[end] = 0;
    return f
end
f = compute_rhs_vector()
@benchmark compute_rhs_vector()

function compute_system_matrix()
    IJV = map(row_partition) do row_indices
        I,J,V = Int[], Int[], Float64[]
        for global_row in local_to_global(row_indices)
            if global_row == 1
                # not needed due to boundary condition: xl, xr = points[global_row,:][1]
                # not needed due to boundary condition: Aloc = (1/(xr-xl))*[1 -1; -1 1]
                # not needed due to boundary condition: push!(I,global_row)
                # not needed due to boundary condition: push!(J,global_row)
                # not needed due to boundary condition: push!(V,Aloc[1,1])
                # not needed due to boundary condition: push!(I,global_row)
                # not needed due to boundary condition: push!(J,global_row+1)
                # not needed due to boundary condition: push!(V,Aloc[1,2])
    #..handle the boundary conditions in the matrix
    # A[1,1] = 1;     A[1,2] = 0
                push!(I,global_row)
                push!(J,global_row)
                push!(V,1.0)
                push!(I,global_row)
                push!(J,global_row+1)
                push!(V,0.0)
            elseif global_row == N+1
                # not needed due to boundary condition: xl, xr = points[global_row-1,:][1]
                # not needed due to boundary condition: Aloc = (1/(xr-xl))*[1 -1; -1 1]
                # not needed due to boundary condition: push!(I,global_row)
                # not needed due to boundary condition: push!(J,global_row-1)
                # not needed due to boundary condition: push!(V,Aloc[2,1])
                # not needed due to boundary condition: push!(I,global_row)
                # not needed due to boundary condition: push!(J,global_row)
                # not needed due to boundary condition: push!(V,Aloc[2,2])
    #..handle the boundary conditions in the matrix
    # A[end,end-1]=0; A[end,end] = 1
                push!(I,global_row)
                push!(J,global_row-1)
                push!(V,0.0)
                push!(I,global_row)
                push!(J,global_row)
                push!(V,1.0)
            else
                xll, xrl = points[global_row-1,:][1]
                Alocl = (1/(xrl-xll))*[1 -1; -1 1]
                xlr, xrr = points[global_row,:][1]
                Alocr = (1/(xrr-xlr))*[1 -1; -1 1]
                push!(I,global_row)
                push!(J,global_row-1)
                push!(V,Alocl[2,1])
                push!(I,global_row)
                push!(J,global_row)
                push!(V,Alocl[2,2] + Alocr[1,1])
                push!(I,global_row)
                push!(J,global_row+1)
                push!(V,Alocr[1,2])
            end
        end
        I,J,V
    end

    I,J,V = tuple_of_arrays(IJV)
    col_partition = row_partition
    A = psparse!(I,J,V,row_partition,col_partition) |> fetch
    return A
end
A = compute_system_matrix()
@benchmark compute_system_matrix()

# Solve
u_part = similar(f,axes(A,2))
u_part .= f
@benchmark IterativeSolvers.cg!(u_part,A,f)

# Check the solution
r = A*u_part - f
norm(r)

#...there must be a better way to do this
u = []
for idx = 1:np
    global u = vcat(u, own_values(u_part)[idx])
end

#..plot the solution  
p1=plot(x,u,shape=:circle,lw=2,legend=false)
xlabel!("x") 
ylabel!("u(x)")
title!("Numerically computed solution")

savefig(p1, "p1.png")
