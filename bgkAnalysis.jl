using StaticArrays;

D2Q9MRTMatrix = SA[ 1  1  1  1  1  1  1  1  1;
                    0 -1 -1 -1  0  1  1  1  0;
                    0  1  0 -1 -1 -1  0  1  1;
                   -4  2 -1  2 -1  2 -1  2 -1;
                    4  1 -2  1 -2  1 -2  1 -2;
                    0 -1  2 -1  0  1 -2  1  0;
                    0  1  0 -1  2 -1  0  1 -2;
                    0  0  1  0 -1  0  1  0 -1;
                    0 -1  0  1  0 -1  0  1  0];

D2Q9MRTMatrixInv = inv(D2Q9MRTMatrix);

D2Q9::Lattice = Lattice(2,9,10,11,12, [[0  0]; [-1  1]; [-1  0]; [-1 -1]; [0 -1]; [1 -1]; [1  0]; [1  1]; [0  1]], [4/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9], 3.0, D2Q9MRTMatrix, D2Q9MRTMatrixInv);

struct Lattice
    d::Int
    q::Int
    rhoIndex::Int
    uIndex::Int 
    dataSize::Int
    c::Matrix{Int64}
    t::Vector{Float64}
    invCs2::Float64
    M::AbstractMatrix{Float64}
    Minv::AbstractMatrix{Float64}
end


function equilibrium(iPop::Int64, rho::Float64, u::AbstractArray{Float64}, uSqr::Float64, lat::Lattice)
    c_u::Float64 = 0.;
    for iD in 1:lat.d
        c_u += lat.c[iPop, iD] * u[iD];
    end
    return rho * lat.t[iPop] * (1. + lat.invCs2*c_u + lat.invCs2^2 * 0.5 * c_u^2 - lat.invCs2 * 0.5 * uSqr) - lat.t[iPop];
end

function bgkCollisionOperation!(data::AbstractArray{Float64}, uWrk::AbstractArray{Float64}, lat::Lattice, omega::Float64) 
    rho::Float64 = 1.0;
    fill!(uWrk, 0.0);

    for iPop in 1:lat.q
        rho += data[iPop];
        for iD in 1:lat.d
            uWrk[iD] += data[iPop] * lat.c[iPop, iD];
        end
    end
    uWrk ./= rho;
    uSqr = dot(uWrk,uWrk);

    data[lat.rhoIndex] = rho;
    for iD in 1:lat.d
        data[lat.uIndex-1+iD] = uWrk[iD];
    end
    
    for iPop in 1:lat.q
        data[iPop] = (data[iPop] * (1.0 - omega)) + omega*equilibrium(iPop, rho, uWrk, uSqr, lat);
    end
    return nothing;
end