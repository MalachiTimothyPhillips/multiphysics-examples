/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#pragma once

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Export.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Kokkos_Core.hpp>
#include <cmath>

#include "multiphys_mesh.hpp"

namespace tt {

using crs_type    = Tpetra::CrsMatrix<SC,LO,GO,NO>;
using vec_type    = Tpetra::Vector<SC,LO,GO,NO>;
using export_type = Tpetra::Export<LO,GO,NO>;
using graph_type  = Tpetra::CrsGraph<LO,GO,NO>;

inline constexpr size_t nnzPerRowQ1Estimate() { return 9; }

inline void shapeQ1(SC xi, SC eta, SC N[4], SC dN_dxi[4], SC dN_deta[4])
{
  N[0] = 0.25*(1-xi)*(1-eta);
  N[1] = 0.25*(1+xi)*(1-eta);
  N[2] = 0.25*(1+xi)*(1+eta);
  N[3] = 0.25*(1-xi)*(1+eta);

  dN_dxi[0]  = -0.25*(1-eta);
  dN_dxi[1]  =  0.25*(1-eta);
  dN_dxi[2]  =  0.25*(1+eta);
  dN_dxi[3]  = -0.25*(1+eta);

  dN_deta[0] = -0.25*(1-xi);
  dN_deta[1] = -0.25*(1+xi);
  dN_deta[2] =  0.25*(1+xi);
  dN_deta[3] =  0.25*(1-xi);
}

inline void invert2x2(const SC J[2][2], SC invJ[2][2], SC& detJ)
{
  detJ = J[0][0]*J[1][1] - J[0][1]*J[1][0];
  const SC invdet = 1.0/detJ;
  invJ[0][0] =  J[1][1]*invdet;
  invJ[0][1] = -J[0][1]*invdet;
  invJ[1][0] = -J[1][0]*invdet;
  invJ[1][1] =  J[0][0]*invdet;
}

inline Teuchos::RCP<graph_type>
buildQ1GraphFromOwnedElements(const Teuchos::RCP<const map_type>& overlapNodeMap,
                             const ConnView& ownedElemConn)
{
  auto G = Teuchos::rcp(new graph_type(overlapNodeMap, nnzPerRowQ1Estimate()));
  Teuchos::Array<GO> cols(4);

  for (size_t e=0; e<ownedElemConn.extent(0); ++e) {
    GO gid[4];
    for (int a=0;a<4;++a) gid[a] = ownedElemConn(e,a);

    for (int i=0;i<4;++i) {
      for (int j=0;j<4;++j) cols[j] = gid[j];
      G->insertGlobalIndices(gid[i], cols());
    }
  }
  G->fillComplete(overlapNodeMap, overlapNodeMap);
  return G;
}

inline void assembleStiffnessAndMassQ1(
    const Teuchos::RCP<const map_type>& overlapNodeMap,
    const CoordView& coordsOverlap,
    const ConnView& ownedElemConn,
    const vec_type& kNodalOverlap,
    const Teuchos::RCP<crs_type>& K_ov,
    const Teuchos::RCP<crs_type>& M_ov)
{
  auto kView = kNodalOverlap.getLocalViewHost(Tpetra::Access::ReadOnly);

  const SC a = 1.0/std::sqrt(3.0);
  const SC gp[2] = {-a, a};

  SC Ke[4][4], Me[4][4];

  for (size_t e=0; e<ownedElemConn.extent(0); ++e) {
    GO gid[4]; LO lid[4];
    for (int aN=0;aN<4;++aN) {
      gid[aN] = ownedElemConn(e,aN);
      lid[aN] = overlapNodeMap->getLocalElement(gid[aN]);
      if (lid[aN] == Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("assembleStiffnessAndMassQ1: element node missing from overlap map");
    }

    SC x[4], y[4];
    for (int aN=0;aN<4;++aN) { x[aN]=coordsOverlap(lid[aN],0); y[aN]=coordsOverlap(lid[aN],1); }

    SC kElem = 0.0;
    for (int aN=0;aN<4;++aN) kElem += kView(lid[aN],0);
    kElem *= 0.25;

    for (int i=0;i<4;++i) for (int j=0;j<4;++j) { Ke[i][j]=0.0; Me[i][j]=0.0; }

    for (int ixi=0; ixi<2; ++ixi) for (int ieta=0; ieta<2; ++ieta) {
      const SC xi=gp[ixi], eta=gp[ieta];
      const SC w = 1.0;

      SC N[4], dN_dxi[4], dN_deta[4];
      shapeQ1(xi,eta,N,dN_dxi,dN_deta);

      SC J[2][2] = {{0,0},{0,0}};
      for (int aN=0;aN<4;++aN) {
        J[0][0] += dN_dxi[aN]*x[aN];
        J[0][1] += dN_deta[aN]*x[aN];
        J[1][0] += dN_dxi[aN]*y[aN];
        J[1][1] += dN_deta[aN]*y[aN];
      }

      SC invJ[2][2], detJ=0.0;
      invert2x2(J, invJ, detJ);

      SC dN_dx[4], dN_dy[4];
      for (int aN=0;aN<4;++aN) {
        dN_dx[aN] = invJ[0][0]*dN_dxi[aN] + invJ[1][0]*dN_deta[aN];
        dN_dy[aN] = invJ[0][1]*dN_dxi[aN] + invJ[1][1]*dN_deta[aN];
      }

      const SC dV = w*detJ;

      for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
        Ke[i][j] += kElem * (dN_dx[i]*dN_dx[j] + dN_dy[i]*dN_dy[j]) * dV;
        Me[i][j] += (N[i]*N[j]) * dV;
      }
    }

    Teuchos::Array<LO> lcols(4);
    Teuchos::Array<SC> valsK(4), valsM(4);
    for (int i=0;i<4;++i) {
      const LO lrow = lid[i];
      for (int j=0;j<4;++j) { lcols[j]=lid[j]; valsK[j]=Ke[i][j]; valsM[j]=Me[i][j]; }
      K_ov->sumIntoLocalValues(lrow, lcols(), valsK());
      M_ov->sumIntoLocalValues(lrow, lcols(), valsM());
    }
  }
}

inline Teuchos::RCP<crs_type>
exportToOwned(const Teuchos::RCP<const crs_type>& Aov,
              const Teuchos::RCP<const map_type>& ownedMap,
              const Teuchos::RCP<const map_type>& overlapMap)
{
  auto Aown = Teuchos::rcp(new crs_type(ownedMap, nnzPerRowQ1Estimate()));
  export_type exporter(overlapMap, ownedMap);
  Aown->doExport(*Aov, exporter, Tpetra::ADD);
  Aown->fillComplete(ownedMap, ownedMap);
  return Aown;
}

struct SubBlocks {
  Teuchos::RCP<crs_type> Ke, Kl, M;
};

inline SubBlocks buildKM_OverlapExported(
    int Nx, int Ny,
    SC x0, SC x1, SC y0, SC y1,
    const std::function<void(vec_type&, const CoordView&)>& fill_ke,
    const std::function<void(vec_type&, const CoordView&)>& fill_kl)
{
  auto comm = Tpetra::getDefaultComm();
  const GO N = GO(Nx)*GO(Ny);
  auto ownedMap = Teuchos::rcp(new map_type(N, 0, comm));

  ConnView ownedElemConn = buildOwnedElementConnectivity(ownedMap, Nx, Ny);
  auto overlapMap = buildOverlapNodeMap(ownedMap, ownedElemConn);
  CoordView coords = buildCoordsStructured(overlapMap, Nx, Ny, x0,x1,y0,y1);

  vec_type ke_ov(overlapMap), kl_ov(overlapMap);
  fill_ke(ke_ov, coords);
  fill_kl(kl_ov, coords);

  auto G_ov  = buildQ1GraphFromOwnedElements(overlapMap, ownedElemConn);
  auto Ke_ov = Teuchos::rcp(new crs_type(G_ov));
  auto Kl_ov = Teuchos::rcp(new crs_type(G_ov));
  auto M_ov  = Teuchos::rcp(new crs_type(G_ov));
  auto dummy = Teuchos::rcp(new crs_type(G_ov));

  assembleStiffnessAndMassQ1(overlapMap, coords, ownedElemConn, ke_ov, Ke_ov, M_ov);
  assembleStiffnessAndMassQ1(overlapMap, coords, ownedElemConn, kl_ov, Kl_ov, dummy);

  Ke_ov->fillComplete(overlapMap, overlapMap);
  Kl_ov->fillComplete(overlapMap, overlapMap);
  M_ov ->fillComplete(overlapMap, overlapMap);

  SubBlocks out;
  out.Ke = exportToOwned(Ke_ov, ownedMap, overlapMap);
  out.Kl = exportToOwned(Kl_ov, ownedMap, overlapMap);
  out.M  = exportToOwned(M_ov,  ownedMap, overlapMap);
  return out;
}

} // namespace tt