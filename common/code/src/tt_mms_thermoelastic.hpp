#pragma once

#include <Tpetra_MultiVector.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <cmath>

#include "tt_mesh.hpp"
#include "tt_q1_assembly.hpp" // shapeQ1/invert2x2

namespace tt::mms_te {

using mv_type = Tpetra::MultiVector<SC,LO,GO,NO>;

constexpr SC pi()
{
  return SC(3.141592653589793238462643383279502884);
}

// ---- Manufactured exact fields ----
inline SC ux(SC x, SC y) { return std::sin(pi()*x)*std::sin(pi()*y); }
inline SC uy(SC x, SC y) { return std::sin(2*pi()*x)*std::sin(pi()*y); }
inline SC T (SC x, SC y) { return std::sin(pi()*x)*std::sin(2*pi()*y); }

// ---- Derivatives we need ----
inline SC dux_dx(SC x, SC y) { return pi()*std::cos(pi()*x)*std::sin(pi()*y); }
inline SC dux_dy(SC x, SC y) { return pi()*std::sin(pi()*x)*std::cos(pi()*y); }
inline SC duy_dx(SC x, SC y) { return 2*pi()*std::cos(2*pi()*x)*std::sin(pi()*y); }
inline SC duy_dy(SC x, SC y) { return pi()*std::sin(2*pi()*x)*std::cos(pi()*y); }

inline SC divu(SC x, SC y) { return dux_dx(x,y) + duy_dy(x,y); }

inline SC ddivu_dx(SC x, SC y)
{
  // d/dx [ pi cos(pi x) sin(pi y) + pi sin(2pi x) cos(pi y) ]
  return -pi()*pi()*std::sin(pi()*x)*std::sin(pi()*y)
         + SC(2)*pi()*pi()*std::cos(2*pi()*x)*std::cos(pi()*y);
}

inline SC ddivu_dy(SC x, SC y)
{
  // d/dy [ pi cos(pi x) sin(pi y) + pi sin(2pi x) cos(pi y) ]
  return pi()*pi()*std::cos(pi()*x)*std::cos(pi()*y)
         - pi()*pi()*std::sin(2*pi()*x)*std::sin(pi()*y);
}

inline SC Lap_ux(SC x, SC y)
{
  // u_x = sin(pi x) sin(pi y) => Δ = -2 pi^2 sin(pi x) sin(pi y)
  return -SC(2)*pi()*pi()*std::sin(pi()*x)*std::sin(pi()*y);
}

inline SC Lap_uy(SC x, SC y)
{
  // u_y = sin(2pi x) sin(pi y) => Δ = -(4+1)pi^2 * sin(2pi x) sin(pi y)
  return -SC(5)*pi()*pi()*std::sin(2*pi()*x)*std::sin(pi()*y);
}

inline SC dT_dx(SC x, SC y) { return pi()*std::cos(pi()*x)*std::sin(2*pi()*y); }
inline SC dT_dy(SC x, SC y) { return SC(2)*pi()*std::sin(pi()*x)*std::cos(2*pi()*y); }

inline SC Lap_T(SC x, SC y)
{
  // T = sin(pi x) sin(2pi y) => Δ = -(1+4)pi^2 * sin(pi x) sin(2pi y)
  return -SC(5)*pi()*pi()*std::sin(pi()*x)*std::sin(2*pi()*y);
}

// ---- Forcings (strong form) ----
// b = -mu Δu - (lambda+mu) ∇(div u) + beta ∇T
inline SC bx(SC x, SC y, SC lambda, SC mu, SC beta)
{
  return -mu*Lap_ux(x,y) - (lambda+mu)*ddivu_dx(x,y) + beta*dT_dx(x,y);
}
inline SC by(SC x, SC y, SC lambda, SC mu, SC beta)
{
  return -mu*Lap_uy(x,y) - (lambda+mu)*ddivu_dy(x,y) + beta*dT_dy(x,y);
}

// q = -kappa ΔT + rhoCp*(w·∇T) - eta*(div u)
inline SC qsrc(SC x, SC y,
               SC kappa, SC rhoCp, SC wx, SC wy,
               SC eta, SC lambda, SC mu, SC beta)
{
  (void)lambda; (void)mu; (void)beta; // not used in q; keep signature flexible
  return -kappa*Lap_T(x,y) + rhoCp*(wx*dT_dx(x,y) + wy*dT_dy(x,y)) - eta*divu(x,y);
}

// Assemble monolithic RHS and exact on overlap monolithic map (3 fields):
// ordering: ux(node), uy(node+N), T(node+2N)
inline void assembleMonolithicRHS_andExact_onOverlap(
    const Teuchos::RCP<const map_type>& overlapNodeMap,
    const Teuchos::RCP<const map_type>& overlapMonoMap,
    const CoordView& coordsOverlap,
    const ConnView& ownedElemConn,
    mv_type& b_ov,
    mv_type& xexact_ov,
    int Nx, int Ny,
    SC lambda, SC mu,
    SC kappa, SC rhoCp, SC wx, SC wy,
    SC beta, SC eta)
{
  const GO NglobNodes = GO(Nx)*GO(Ny);

  if (b_ov.getNumVectors()!=1 || xexact_ov.getNumVectors()!=1)
    throw std::runtime_error("MMS thermoelastic: expect b and xexact to have 1 vector");
  if (!b_ov.getMap()->isSameAs(*overlapMonoMap) || !xexact_ov.getMap()->isSameAs(*overlapMonoMap))
    throw std::runtime_error("MMS thermoelastic: map mismatch");

  // ---- nodal exact ----
  xexact_ov.putScalar(0.0);
  {
    auto X = xexact_ov.getLocalViewHost(Tpetra::Access::OverwriteAll);
    const LO nLocalNodes = (LO)overlapNodeMap->getLocalNumElements();
    for (LO l=0; l<nLocalNodes; ++l) {
      const GO nodeG = overlapNodeMap->getGlobalElement(l);
      const SC x = coordsOverlap(l,0), y = coordsOverlap(l,1);

      const LO lr_x = overlapMonoMap->getLocalElement(nodeG);
      const LO lr_y = overlapMonoMap->getLocalElement(nodeG + NglobNodes);
      const LO lr_T = overlapMonoMap->getLocalElement(nodeG + 2*NglobNodes);

      if (lr_x==Teuchos::OrdinalTraits<LO>::invalid() ||
          lr_y==Teuchos::OrdinalTraits<LO>::invalid() ||
          lr_T==Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("MMS thermoelastic: mono overlap map missing dof");

      X(lr_x,0) = ux(x,y);
      X(lr_y,0) = uy(x,y);
      X(lr_T,0) = T (x,y);
    }
  }

  // ---- quadrature RHS ----
  b_ov.putScalar(0.0);

  const SC a = 1.0/std::sqrt(3.0);
  const SC gp[2] = {-a, a};

  for (size_t e=0; e<ownedElemConn.extent(0); ++e) {
    GO gid[4]; LO lid[4];
    for (int aN=0;aN<4;++aN) {
      gid[aN] = ownedElemConn(e,aN);
      lid[aN] = overlapNodeMap->getLocalElement(gid[aN]);
      if (lid[aN] == Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("MMS thermoelastic: element node missing from overlap node map");
    }

    SC x[4], y[4];
    for (int aN=0;aN<4;++aN) { x[aN]=coordsOverlap(lid[aN],0); y[aN]=coordsOverlap(lid[aN],1); }

    SC fx[4] = {0,0,0,0};
    SC fy[4] = {0,0,0,0};
    SC fT[4] = {0,0,0,0};

    for (int ixi=0; ixi<2; ++ixi) for (int ieta=0; ieta<2; ++ieta) {
      const SC xi=gp[ixi], eta=gp[ieta];
      const SC wq = 1.0;

      SC N[4], dN_dxi[4], dN_deta[4];
      tt::shapeQ1(xi,eta,N,dN_dxi,dN_deta);

      SC xq=0.0,yq=0.0;
      for (int aN=0;aN<4;++aN) { xq += N[aN]*x[aN]; yq += N[aN]*y[aN]; }

      SC J[2][2] = {{0,0},{0,0}};
      for (int aN=0;aN<4;++aN) {
        J[0][0] += dN_dxi[aN]*x[aN];
        J[0][1] += dN_deta[aN]*x[aN];
        J[1][0] += dN_dxi[aN]*y[aN];
        J[1][1] += dN_deta[aN]*y[aN];
      }

      SC invJ[2][2], detJ=0.0;
      tt::invert2x2(J, invJ, detJ);
      const SC dV = wq*detJ;

      const SC bxq = bx(xq,yq, lambda, mu, beta);
      const SC byq = by(xq,yq, lambda, mu, beta);
      const SC qq  = qsrc(xq,yq, kappa, rhoCp, wx, wy, eta, lambda, mu, beta);

      for (int i=0;i<4;++i) {
        fx[i] += bxq * N[i] * dV;
        fy[i] += byq * N[i] * dV;
        fT[i] += qq  * N[i] * dV;
      }
    }

    for (int aN=0;aN<4;++aN) {
      const GO nodeG = gid[aN];

      const LO lr_x = overlapMonoMap->getLocalElement(nodeG);
      const LO lr_y = overlapMonoMap->getLocalElement(nodeG + NglobNodes);
      const LO lr_T = overlapMonoMap->getLocalElement(nodeG + 2*NglobNodes);

      if (lr_x==Teuchos::OrdinalTraits<LO>::invalid() ||
          lr_y==Teuchos::OrdinalTraits<LO>::invalid() ||
          lr_T==Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("MMS thermoelastic: scatter dof missing from mono overlap map");

      b_ov.sumIntoLocalValue(lr_x, 0, fx[aN]);
      b_ov.sumIntoLocalValue(lr_y, 0, fy[aN]);
      b_ov.sumIntoLocalValue(lr_T, 0, fT[aN]);
    }
  }
}

} // namespace tt::mms_te