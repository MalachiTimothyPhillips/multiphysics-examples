#pragma once

#include <Tpetra_MultiVector.hpp>
#include <Teuchos_RCP.hpp>
#include <cmath>

#include "tt_mesh.hpp"
#include "tt_q1_assembly.hpp" // for shapeQ1/invert2x2

namespace tt::mms {

using mv_type = Tpetra::MultiVector<SC,LO,GO,NO>;

inline SC Te(SC x, SC y)
{
  constexpr SC pi = 3.141592653589793238462643383279502884;
  return std::sin(pi*x)*std::sin(pi*y);
}

inline SC Tl(SC x, SC y)
{
  constexpr SC pi = 3.141592653589793238462643383279502884;
  return std::sin(2*pi*x)*std::sin(pi*y);
}

inline void assembleMonolithicRHS_andExact_onOverlap(
    const Teuchos::RCP<const map_type>& overlapNodeMap,
    const Teuchos::RCP<const map_type>& overlapMonoMap, // blocked [Te nodes][Tl nodes]
    const CoordView& coordsOverlap,
    const ConnView& ownedElemConn,
    mv_type& b_ov,
    mv_type& xexact_ov,
    int Nx, int Ny,
    SC ke, SC kl, SC g)
{
  constexpr SC pi = 3.141592653589793238462643383279502884;
  const GO NglobNodes = GO(Nx)*GO(Ny);

  if (b_ov.getNumVectors()!=1 || xexact_ov.getNumVectors()!=1)
    throw std::runtime_error("MMS: expect b and xexact to have 1 vector");
  if (!b_ov.getMap()->isSameAs(*overlapMonoMap) || !xexact_ov.getMap()->isSameAs(*overlapMonoMap))
    throw std::runtime_error("MMS: map mismatch");

  // nodal exact
  xexact_ov.putScalar(0.0);
  {
    auto X = xexact_ov.getLocalViewHost(Tpetra::Access::OverwriteAll);
    const LO nLocalNodes = (LO)overlapNodeMap->getLocalNumElements();
    for (LO l=0; l<nLocalNodes; ++l) {
      const GO nodeG = overlapNodeMap->getGlobalElement(l);
      const SC x = coordsOverlap(l,0), y = coordsOverlap(l,1);

      const LO lr_e = overlapMonoMap->getLocalElement(nodeG);
      const LO lr_l = overlapMonoMap->getLocalElement(nodeG + NglobNodes);
      if (lr_e==Teuchos::OrdinalTraits<LO>::invalid() || lr_l==Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("MMS: mono overlap map does not contain expected dof");

      X(lr_e,0) = Te(x,y);
      X(lr_l,0) = Tl(x,y);
    }
  }

  // quadrature RHS
  b_ov.putScalar(0.0);

  const SC a = 1.0/std::sqrt(3.0);
  const SC gp[2] = {-a, a};

  for (size_t e=0; e<ownedElemConn.extent(0); ++e) {
    GO gid[4]; LO lid[4];
    for (int aN=0;aN<4;++aN) {
      gid[aN] = ownedElemConn(e,aN);
      lid[aN] = overlapNodeMap->getLocalElement(gid[aN]);
      if (lid[aN] == Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("MMS: element node missing from overlap node map");
    }

    SC x[4], y[4];
    for (int aN=0;aN<4;++aN) { x[aN]=coordsOverlap(lid[aN],0); y[aN]=coordsOverlap(lid[aN],1); }

    SC fe[4] = {0,0,0,0};
    SC fl[4] = {0,0,0,0};

    for (int ixi=0; ixi<2; ++ixi) for (int ieta=0; ieta<2; ++ieta) {
      const SC xi=gp[ixi], eta=gp[ieta];
      const SC w = 1.0;

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
      const SC dV = w*detJ;

      const SC Teq = std::sin(pi*xq)*std::sin(pi*yq);
      const SC Tlq = std::sin(2*pi*xq)*std::sin(pi*yq);

      const SC Qe = (2.0*pi*pi*ke)*Teq + g*(Teq - Tlq);
      const SC Ql = (5.0*pi*pi*kl)*Tlq + g*(Tlq - Teq);

      for (int i=0;i<4;++i) { fe[i] += Qe*N[i]*dV; fl[i] += Ql*N[i]*dV; }
    }

    for (int aN=0;aN<4;++aN) {
      const GO nodeG = gid[aN];

      const LO lr_e = overlapMonoMap->getLocalElement(nodeG);
      const LO lr_l = overlapMonoMap->getLocalElement(nodeG + NglobNodes);
      if (lr_e==Teuchos::OrdinalTraits<LO>::invalid() || lr_l==Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("MMS: scatter dof missing from mono overlap map");

      b_ov.sumIntoLocalValue(lr_e, 0, fe[aN]);
      b_ov.sumIntoLocalValue(lr_l, 0, fl[aN]);
    }
  }
}

} // namespace tt::mms