// tt_pnp_linear_q1.hpp
#pragma once

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Export.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <vector>
#include <stdexcept>

#include "tt_mesh.hpp"
#include "tt_q1_assembly.hpp"
#include "tt_dof.hpp"
#include "tt_transfer.hpp"

namespace tt::pnp_linear {

using crs_type = tt::crs_type;
using mv_type  = Tpetra::MultiVector<SC,LO,GO,NO>;
using export_type = tt::export_type;
using vec_type = tt::vec_type;

struct Params {
  SC eps   = 1.0;
  SC D     = 1.0;
  SC alpha = 1.0; // linear drift strength
  SC zF    = 1.0; // coupling in Poisson
};

// Nonhomogeneous Dirichlet for monolithic N-field with field-blocked-by-offset layout
template<class ValueFunc>
inline void applyDirichlet_MonolithicNField_SetValue(
    Tpetra::CrsMatrix<SC,LO,GO,NO>& A,
    mv_type* b, // nullable
    int Nx, int Ny,
    int nFields,
    GO NglobNodes,
    const ValueFunc& valueAtBoundary /* valueAtBoundary(field,nodeGID)->SC */)
{
  if (!A.isFillComplete()) throw std::runtime_error("Dirichlet: A must be fillComplete()");
  auto rowMap = A.getRowMap();
  auto colMap = A.getColMap();

  A.resumeFill();

  const LO nLocalRows = (LO)rowMap->getLocalNumElements();
  for (LO lrow=0; lrow<nLocalRows; ++lrow) {
    const GO grow = rowMap->getGlobalElement(lrow);
    const GO nodeG = grow % NglobNodes;
    const int field = int(grow / NglobNodes);

    if (field < 0 || field >= nFields) continue;
    if (!tt::isBoundaryNode(nodeG, Nx, Ny)) continue;

    const size_t nnz = A.getNumEntriesInLocalRow(lrow);

    typename Tpetra::CrsMatrix<SC,LO,GO,NO>::nonconst_local_inds_host_view_type inds("inds", nnz);
    typename Tpetra::CrsMatrix<SC,LO,GO,NO>::nonconst_values_host_view_type vals("vals", nnz);

    size_t n=0;
    A.getLocalRowCopy(lrow, inds, vals, n);

    const LO ldiag = colMap->getLocalElement(grow);
    for (size_t k=0; k<n; ++k) vals(k) = (inds(k)==ldiag ? SC(1) : SC(0));

    A.replaceLocalValues(lrow, n, vals.data(), inds.data());

    if (b) {
      const SC g = valueAtBoundary(field, nodeG);
      for (LO j=0; j<(LO)b->getNumVectors(); ++j)
        b->replaceLocalValue(lrow, j, g);
    }
  }

  A.fillComplete(A.getDomainMap(), A.getRangeMap());
}

// Assemble element matrices and element load vectors for the linear system.
//
// Weak forms (Galerkin):
//
// Poisson:    ∫ eps ∇phi·∇v  - ∫ zF c v  = ∫ rho_f v
//
// NP linear drift:
//   With flux F = D∇c + alpha ∇phi - u c  (so that -F = -D∇c - alpha∇phi + u c as above)
//   ∫ F·∇w = ∫ s w
//
// This matches taking div(-D∇c - alpha∇phi + u c) = s and integrating by parts once.
//
// Note: u is prescribed and enters as convection-like term in c-equation.
inline void assemble_element_Q1(
    const SC x[4], const SC y[4],
    const Params& p,
    const SC rhoFN[4], const SC sNPN[4],
    const SC uNx[4], const SC uNy[4],
    // outputs
    SC Aphi_phi[4][4], SC Aphi_c[4][4],
    SC Ac_phi [4][4], SC Ac_c [4][4],
    SC fphi[4], SC fc[4])
{
  for (int i=0;i<4;++i) {
    fphi[i]=0.0; fc[i]=0.0;
    for (int j=0;j<4;++j) {
      Aphi_phi[i][j]=0.0; Aphi_c[i][j]=0.0;
      Ac_phi [i][j]=0.0; Ac_c [i][j]=0.0;
    }
  }

  const SC a = 1.0/std::sqrt(3.0);
  const SC gp[2] = {-a, a};

  for (int ixi=0; ixi<2; ++ixi) for (int ieta=0; ieta<2; ++ieta) {
    const SC xi=gp[ixi], eta=gp[ieta];
    const SC wq = 1.0;

    SC N[4], dN_dxi[4], dN_deta[4];
    tt::shapeQ1(xi,eta,N,dN_dxi,dN_deta);

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

    SC dN_dx[4], dN_dy[4];
    for (int aN=0;aN<4;++aN) {
      dN_dx[aN] = invJ[0][0]*dN_dxi[aN] + invJ[1][0]*dN_deta[aN];
      dN_dy[aN] = invJ[0][1]*dN_dxi[aN] + invJ[1][1]*dN_deta[aN];
    }

    // interpolate sources and velocity
    SC rhoF_q=0.0, sNP_q=0.0, ux_q=0.0, uy_q=0.0;
    for (int aN=0;aN<4;++aN) {
      rhoF_q += N[aN]*rhoFN[aN];
      sNP_q  += N[aN]*sNPN[aN];
      ux_q   += N[aN]*uNx[aN];
      uy_q   += N[aN]*uNy[aN];
    }

    for (int i=0;i<4;++i) {
      // RHS loads
      fphi[i] += rhoF_q * N[i] * dV;
      fc[i]   += sNP_q  * N[i] * dV;

      for (int j=0;j<4;++j) {
        // Poisson stiffness
        Aphi_phi[i][j] += p.eps*(dN_dx[j]*dN_dx[i] + dN_dy[j]*dN_dy[i]) * dV;

        // coupling -∫ zF c v
        Aphi_c[i][j] += (-p.zF) * (N[j]*N[i]) * dV;

        // NP: ∫ (D∇c + alpha ∇phi - u c) · ∇w
        // Ac_c: diffusion + convection(-u c)
        Ac_c[i][j] += p.D*(dN_dx[j]*dN_dx[i] + dN_dy[j]*dN_dy[i]) * dV;
        Ac_c[i][j] += (-ux_q) * (N[j]*dN_dx[i]) * dV;
        Ac_c[i][j] += (-uy_q) * (N[j]*dN_dy[i]) * dV;

        // Ac_phi: drift coupling (alpha ∇phi) · ∇w
        Ac_phi[i][j] += p.alpha*(dN_dx[j]*dN_dx[i] + dN_dy[j]*dN_dy[i]) * dV;
      }
    }
  }
}

// Build A and b on owned monolithic map, plus xexact on owned monolithic map.
// fill_onOverlap provides MMS nodal values for: phi_ex, c_ex, rho_f, s, u.
struct SystemAndExact {
  Teuchos::RCP<crs_type> A;
  Teuchos::RCP<mv_type>  b;
  Teuchos::RCP<mv_type>  xexact;
};

inline SystemAndExact buildLinearPNP_MMS_System_OverlapExported(
    int Nx, int Ny, SC x0, SC x1, SC y0, SC y1,
    const Params& p,
    const std::function<void(const CoordView&,
                             vec_type& phi_ex, vec_type& c_ex,
                             vec_type& rho_f, vec_type& s_np,
                             vec_type& ux, vec_type& uy)>& fill_onOverlap)
{
  auto comm = Tpetra::getDefaultComm();
  const GO NglobNodes = GO(Nx)*GO(Ny);

  // owned node map
  auto ownedNodeMap = Teuchos::rcp(new map_type(NglobNodes, 0, comm));

  // overlap connectivity/coords
  ConnView ownedElemConn = tt::buildOwnedElementConnectivity(ownedNodeMap, Nx, Ny);
  auto overlapNodeMap = tt::buildOverlapNodeMap(ownedNodeMap, ownedElemConn);
  CoordView coordsOverlap = tt::buildCoordsStructured(overlapNodeMap, Nx, Ny, x0,x1,y0,y1);

  // overlap nodal fields
  vec_type phi_ex(overlapNodeMap), c_ex(overlapNodeMap);
  vec_type rho_f(overlapNodeMap), s_np(overlapNodeMap);
  vec_type uxN(overlapNodeMap), uyN(overlapNodeMap);
  fill_onOverlap(coordsOverlap, phi_ex, c_ex, rho_f, s_np, uxN, uyN);

  // monolithic maps: 2 fields
  const int nFields = 2;
  auto overlapMonoMap = tt::buildMonolithicMapNFieldFromNodeMap(overlapNodeMap, NglobNodes, nFields);
  auto ownedMonoMap   = tt::buildMonolithicMapNFieldFromNodeMap(ownedNodeMap,   NglobNodes, nFields);

  // assemble overlap A and b
  const size_t nnzPerRow = 18;
  auto A_ov = Teuchos::rcp(new crs_type(overlapMonoMap, nnzPerRow));
  A_ov->resumeFill();

  mv_type b_ov(overlapMonoMap, 1);
  b_ov.putScalar(0.0);

  auto rhoV = rho_f.getLocalViewHost(Tpetra::Access::ReadOnly);
  auto sV   = s_np.getLocalViewHost(Tpetra::Access::ReadOnly);
  auto uxV  = uxN.getLocalViewHost(Tpetra::Access::ReadOnly);
  auto uyV  = uyN.getLocalViewHost(Tpetra::Access::ReadOnly);

  for (size_t e=0; e<ownedElemConn.extent(0); ++e) {
    GO gid[4]; LO lid[4];
    for (int a=0;a<4;++a) {
      gid[a] = ownedElemConn(e,a);
      lid[a] = overlapNodeMap->getLocalElement(gid[a]);
      if (lid[a] == Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("PNP linear MMS: element node missing from overlap map");
    }

    SC x[4], y[4], rhoFN[4], sNPN[4], uNx[4], uNy[4];
    for (int a=0;a<4;++a) {
      x[a] = coordsOverlap(lid[a],0);
      y[a] = coordsOverlap(lid[a],1);
      rhoFN[a] = rhoV(lid[a],0);
      sNPN[a]  = sV(lid[a],0);
      uNx[a]   = uxV(lid[a],0);
      uNy[a]   = uyV(lid[a],0);
    }

    SC Aphi_phi[4][4], Aphi_c[4][4], Ac_phi[4][4], Ac_c[4][4];
    SC fphi[4], fc[4];
    assemble_element_Q1(x,y,p, rhoFN,sNPN,uNx,uNy, Aphi_phi,Aphi_c,Ac_phi,Ac_c, fphi,fc);

    // global dof gids for element
    GO phiG[4], cG[4];
    for (int a=0;a<4;++a) {
      phiG[a] = gid[a];
      cG[a]   = gid[a] + NglobNodes;
    }

    // Insert 4x4 blocks via sumIntoGlobalValues
    for (int i=0;i<4;++i) {
      // Poisson row i
      for (int j=0;j<4;++j) {
        A_ov->sumIntoGlobalValues(phiG[i], Teuchos::arrayView(&phiG[j],1), Teuchos::arrayView(&Aphi_phi[i][j],1));
        A_ov->sumIntoGlobalValues(phiG[i], Teuchos::arrayView(&cG[j],1),   Teuchos::arrayView(&Aphi_c[i][j],1));
      }
      b_ov.sumIntoGlobalValue(phiG[i], 0, fphi[i]);

      // NP row i
      for (int j=0;j<4;++j) {
        A_ov->sumIntoGlobalValues(cG[i], Teuchos::arrayView(&phiG[j],1), Teuchos::arrayView(&Ac_phi[i][j],1));
        A_ov->sumIntoGlobalValues(cG[i], Teuchos::arrayView(&cG[j],1),   Teuchos::arrayView(&Ac_c[i][j],1));
      }
      b_ov.sumIntoGlobalValue(cG[i], 0, fc[i]);
    }
  }

  A_ov->fillComplete(overlapMonoMap, overlapMonoMap);

  // export overlap -> owned
  auto A_owned = Teuchos::rcp(new crs_type(ownedMonoMap, nnzPerRow));
  export_type exporter(overlapMonoMap, ownedMonoMap);
  A_owned->doExport(*A_ov, exporter, Tpetra::ADD);
  A_owned->fillComplete(ownedMonoMap, ownedMonoMap);

  auto b_owned = Teuchos::make_rcp<mv_type>(ownedMonoMap, 1);
  tt::exportMonolithicVector(b_ov, *b_owned, *overlapMonoMap, *ownedMonoMap, Tpetra::ADD);

  // build xexact on owned monolithic map (by exporting overlap exact nodal data)
  mv_type xex_ov(overlapMonoMap, 1);
  xex_ov.putScalar(0.0);
  {
    auto xh = xex_ov.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto ph = phi_ex.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto ch = c_ex.getLocalViewHost(Tpetra::Access::ReadOnly);

    const LO nLocalNodes = (LO)overlapNodeMap->getLocalNumElements();
    for (LO l=0; l<nLocalNodes; ++l) {
      const GO nodeG = overlapNodeMap->getGlobalElement(l);
      const LO lphi = overlapMonoMap->getLocalElement(nodeG);
      const LO lc   = overlapMonoMap->getLocalElement(nodeG + NglobNodes);
      xh(lphi,0) = ph(l,0);
      xh(lc,0)   = ch(l,0);
    }
  }

  auto xexact_owned = Teuchos::make_rcp<mv_type>(ownedMonoMap, 1);
  tt::exportMonolithicVector(xex_ov, *xexact_owned, *overlapMonoMap, *ownedMonoMap, Tpetra::INSERT);

  // Dirichlet BCs: set boundary values to exact solution
  // field 0 = phi, field 1 = c
  auto valueAtBoundary = [&](int field, GO nodeG) -> SC {
    // nodeG is global node id; recover coords analytically for structured grid:
    // since coords are available in your buildCoordsStructured on ownedNodeMap, you could also pass coords in.
    // Here assume [0,1]x[0,1] structured:
    int i,j; tt::nodeIJ(nodeG, Nx, i, j);
    const SC X = x0 + (x1-x0)*SC(i)/SC(Nx-1);
    const SC Y = y0 + (y1-y0)*SC(j)/SC(Ny-1);
    return (field==0 ? tt::mms_pnp_linear::phi_ex(X,Y)
                     : tt::mms_pnp_linear::c_ex(X,Y));
  };

  applyDirichlet_MonolithicNField_SetValue(*A_owned, b_owned.get(), Nx, Ny, nFields, NglobNodes, valueAtBoundary);

  return {A_owned, b_owned, xexact_owned};
}

} // namespace tt::pnp_linear