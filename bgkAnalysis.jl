using StaticArrays;
using LinearAlgebra;
using Plots;

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

function computeJacobian!(Jij::AbstractArray{Float64}, inputPops::AbstractArray{Float64}, outputPopsUpper::AbstractArray{Float64}, outputPopsLower::AbstractArray{Float64}, uWrk::AbstractArray{Float64}, lat::Lattice, collisionFunction!, omega::Float64)

    dPop::Float64 = 1.0E-6;
    for j in 1:lat.q
        outputPopsUpper .= inputPops;
        outputPopsUpper[j] += dPop;
        outputPopsLower .= inputPops;
        outputPopsLower[j] -= dPop;

        collisionFunction!(outputPopsUpper, uWrk, lat, omega);
        collisionFunction!(outputPopsLower, uWrk, lat, omega);

        @view(Jij[:, j]) .= (outputPopsUpper[1:lat.q]-outputPopsLower[1:lat.q])./(2*dPop);
    end
end

function eigenvalueDecomp!(eigenvalues::AbstractArray{ComplexF64}, nsEigs::AbstractArray{ComplexF64}, indicator::AbstractArray{Int64}, Jij_comp::AbstractArray{ComplexF64}, k::AbstractArray{Float64}, lat::Lattice, rhoBar::Float64, uBar::Vector{Float64}, desiredViscosity::Float64)
    for i in 1:lat.q
        kDotC = dot(k, lat.c[i, :]);
        Jij_comp[i, :] .*= exp(im*kDotC);
    end

    eigenDecomp = eigen(Jij_comp);
    eigenvalues .= log.(eigenDecomp.values)./im;

    NSMat = complex(zeros(3,3));
    NSMat[1, :] = [dot(k, uBar) rhoBar*k[1] rhoBar*k[2]];
    NSMat[2, :] = [k[1]*(1/lat.invCs2)/rhoBar dot(k,uBar) + im*desiredViscosity*(dot(k,k) + k[1]^2) im*desiredViscosity*k[1]*k[2]];
    NSMat[3, :] = [k[2]*(1/lat.invCs2)/rhoBar im*desiredViscosity*k[1]*k[2] dot(k,uBar) + im*desiredViscosity*(dot(k,k) + k[2]^2)];

    nsEigenDecomp = eigen(NSMat);
    nsEigs .= nsEigenDecomp.values;

    for i in 1:lat.q
        rhoFluctuation = sum(eigenDecomp.vectors[:, i]);
        momFluctuation = sum([eigenDecomp.vectors[:, i] .* lat.c[:, 1] eigenDecomp.vectors[:, i] .* lat.c[:, 2]], dims=1)';

        uFluctuation = (momFluctuation - rhoFluctuation*uBar)/rhoBar;

        Vp = [rhoFluctuation; uFluctuation];

        R = inv(nsEigenDecomp.vectors)*Vp;
        R = norm.(R);
        threshold = 0.99;
        nonObservableThreshold = 0.001;
        if norm(R) < nonObservableThreshold
            indicator[i] = -3; #non observable
        elseif R[1] > threshold
            indicator[i] = -1; #negative acoustic 
        elseif R[2] > threshold
            indicator[i] = 0; #shear
        elseif R[3] > threshold
            indicator[i] = 1; #positive acoustic
        else
            indicator[i] = -2; # non physical
        end
    end
end

rhoBar = 1.0;
uBar = [0.1*(1/sqrt(D2Q9.invCs2)), 0.];
uSqr = dot(uBar, uBar);

inputPops = zeros(D2Q9.dataSize,1);
inputPops[1:D2Q9.q] = broadcast(x->equilibrium(x, rhoBar, uBar, uSqr, D2Q9), 1:D2Q9.q);
inputPops[D2Q9.rhoIndex] = rhoBar;
inputPops[D2Q9.uIndex:D2Q9.uIndex+D2Q9.d-1] = uBar;
outputPopsUpper = copy(inputPops);
outputPopsLower = copy(inputPops);
uWrk = zeros(2,1);

desiredViscosity = 10.0^-6;
tau = desiredViscosity*3 + 0.5;
# tau = 1E-5 + 0.5;
omega = 1/tau;

Jij = zeros(D2Q9.q, D2Q9.q);
computeJacobian!(Jij, inputPops, outputPopsUpper, outputPopsLower, uWrk, D2Q9, bgkCollisionOperation!, omega);

nEig = 50;
kx = LinRange(0, pi, nEig);

eigOmega = complex(zeros(D2Q9.q, nEig));
Jij_comp = complex(Jij);

nsEigOmega = complex(zeros(3, nEig));

indicator = zeros(Int64, D2Q9.q, nEig); #0 --> shear wave, -1: left acoustic, +1: right acoustic, -2:: non physical, -3: non observable

for i in 1:nEig
    Jij_comp .= complex(Jij);
    eigenvalueDecomp!(@view(eigOmega[:, i]), @view(nsEigOmega[:, i]), @view(indicator[:, i]), Jij_comp, [kx[i], 0.], D2Q9, rhoBar, uBar, desiredViscosity)
end

eigOmegaReal = real(eigOmega);
eigOmegaImag = imag(eigOmega);

omegaRealPlot = plot();
for i in 1:D2Q9.q
    for j in 1:nEig
        if indicator[i, j] == -3
            scatter!([kx[j]], [eigOmegaReal[i, j]], markershape=:diamond, markercolor=:grey, markeralpha=0.2, legend=false);
        elseif indicator[i, j] == -2
            scatter!([kx[j]], [eigOmegaReal[i, j]], markershape=:x, markercolor=:black, legend=false);
        elseif indicator[i, j] == -1
            scatter!([kx[j]], [eigOmegaReal[i, j]], markershape=:ltriangle, markercolor=:green, legend=false);
        elseif indicator[i, j] == 0
            scatter!([kx[j]], [eigOmegaReal[i, j]], markershape=:circle, markercolor=:red, legend=false);
        elseif indicator[i, j] == 1
            scatter!([kx[j]], [eigOmegaReal[i, j]], markershape=:rtriangle, markercolor=:blue, legend=false);
        end
    end
end
xlims!(0, pi);
ylims!(-1.2*pi, 1.2*pi);
xlabel!("k_x");
ylabel!("ω_r");

omegaImagPlot = plot();
for i in 1:D2Q9.q
    for j in 1:nEig
        if indicator[i, j] == -3
            scatter!([kx[j]], [eigOmegaImag[i, j]/(-desiredViscosity)], markershape=:diamond, markercolor=:grey, markeralpha=0.2, legend=false);
        elseif indicator[i, j] == -2
            scatter!([kx[j]], [eigOmegaImag[i, j]/(-desiredViscosity)], markershape=:x, markercolor=:black, legend=false);
        elseif indicator[i, j] == -1
            scatter!([kx[j]], [eigOmegaImag[i, j]/(-desiredViscosity)], markershape=:ltriangle, markercolor=:green, legend=false);
        elseif indicator[i, j] == 0
            scatter!([kx[j]], [eigOmegaImag[i, j]/(-desiredViscosity)], markershape=:circle, markercolor=:red, legend=false);
        elseif indicator[i, j] == 1
            scatter!([kx[j]], [eigOmegaImag[i, j]/(-desiredViscosity)], markershape=:rtriangle, markercolor=:blue, legend=false);
        end
    end
end
xlims!(0, pi);
ylims!(-14, 0.1);
xlabel!("k_x");
ylabel!("ω_i/ν");

nsEigOmegaReal = real(nsEigOmega);
nsEigOmegaImag = imag(nsEigOmega);

nsOmegaRealPlot = plot();
for i in 1:3
    scatter!(kx, nsEigOmegaReal[i, :]);
end
xlims!(0, pi);
ylims!(-1.2*pi, 1.2*pi);
xlabel!("k_x");
ylabel!("ω_r");

nsOmegaImagPlot = plot();
for i in 1:3
    scatter!(kx, nsEigOmegaImag[i, :]./(-desiredViscosity));
end
xlims!(0, pi);
ylims!(-14, 0.1);
xlabel!("k_x");
ylabel!("ω_i/ν");


println("done");