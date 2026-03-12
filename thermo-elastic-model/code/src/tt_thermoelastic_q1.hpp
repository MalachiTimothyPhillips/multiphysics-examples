#pragma once

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Export.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <cmath>
#include <vector>

#include "tt_mesh.hpp"
#include "tt_q1_assembly.hpp"

#include "tt_dof.hpp"
#include "tt_dirichlet_blocks.hpp"

namespace tt::thermoelastic {

using crs_type    = tt::crs_type;
using export_type = tt::export_type;

// ---------------- Parameters ----------------
struct Params {
  // elasticity (Lamé)
  SC lambda = 1.0;
  SC mu     = 1.0;

  // heat
  SC kappa  = 1.0;  // conductivity
  SC rhoCp  = 1.0;  // rho*c_p
  SC wx     = 0.0;  // advection velocity
  SC wy     = 0.0;

  // coupling
  SC beta = 1e-2;  // mechanics <- T (stress term -beta T I)
  SC eta  = 1e-2;  // heat <- div(u) (rhs + eta div(u))
};

// ---------------- Output blocks ----------------
//
// Scalar node-based blocks are all N x N (node dofs).
// Additionally, we explicitly assemble:
//  - Auu (2N x 2N) on uMap=[node, node+NglobNodes]
//  - AuT (2N x N)
//  - ATu (N x 2N)
//
// These explicit monolithic blocks avoid Xpetra/Thyra merge/map-mode issues
// and guarantee consistent row maps when you build a 2x2 field split [u;T].
struct BlocksOwned {
  // mechanics (scalar blocks)
  Teuchos::RCP<crs_type> Kxx, Kxy, Kyx, Kyy;

  // heat
  Teuchos::RCP<crs_type> AT;

  // couplings (scalar blocks)
  Teuchos::RCP<crs_type> KxT, KyT; // u <- T
  Teuchos::RCP<crs_type> KTx, KTy; // T <- u

  // explicitly assembled monolithic blocks
  Teuchos::RCP<crs_type> Auu; // 2N x 2N
  Teuchos::RCP<crs_type> AuT; // 2N x N
  Teuchos::RCP<crs_type> ATu; // N x 2N
};

// ---------------- Internal helpers ----------------
inline Teuchos::RCP<tt::graph_type>
buildNodeGraph(const Teuchos::RCP<const map_type>& overlapNodeMap,
               const ConnView& ownedElemConn)
{
  return tt::buildQ1GraphFromOwnedElements(overlapNodeMap, ownedElemConn);
}

inline Teuchos::RCP<crs_type>
exportToOwned(const Teuchos::RCP<const crs_type>& Aov,
              const Teuchos::RCP<const map_type>& ownedMap,
              const Teuchos::RCP<const map_type>& overlapMap)
{
  return tt::exportToOwned(Aov, ownedMap, overlapMap);
}

inline size_t nnzPerRowUQ1Estimate()   { return 18; } // 2 components x ~9 neighbors
inline size_t nnzPerRowRectEstimate() { return 18; }

// Copy a distributed scalar matrix A into B, shifting row/col GIDs by offsets.
// Uses sumIntoGlobalValues so B can accumulate from multiple blocks.
template<class SC_, class LO_, class GO_, class NO_>
void addShiftedBlockInto(Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& B,
                         const Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& A,
                         GO_ rowOffset, GO_ colOffset,
                         SC_ scale = SC_(1))
{
  auto rowMapA = A.getRowMap();
  auto colMapA = A.getColMap();

  const LO_ nLocalRows = (LO_)rowMapA->getLocalNumElements();

  for (LO_ lrow = 0; lrow < nLocalRows; ++lrow) {
    const GO_ growA = rowMapA->getGlobalElement(lrow);
    const GO_ growB = growA + rowOffset;

    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::local_inds_host_view_type inds;
    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::values_host_view_type     vals;
    A.getLocalRowView(lrow, inds, vals);

    const size_t n = inds.extent(0);
    if (n == 0) continue;

    std::vector<GO_> cols(n);
    std::vector<SC_> v(n);

    for (size_t k=0; k<n; ++k) {
      const LO_ lcolA = inds(k);
      const GO_ gcolA = colMapA->getGlobalElement(lcolA);
      cols[k] = gcolA + colOffset;
      v[k]    = scale * vals(k);
    }

    B.sumIntoGlobalValues(growB, (LO_)n, v.data(), cols.data());
  }
}

inline Teuchos::RCP<crs_type>
buildAuu_from_scalar_blocks(const Teuchos::RCP<const map_type>& ownedNodeMap,
                            GO NglobNodes,
                            const Teuchos::RCP<crs_type>& Kxx,
                            const Teuchos::RCP<crs_type>& Kxy,
                            const Teuchos::RCP<crs_type>& Kyx,
                            const Teuchos::RCP<crs_type>& Kyy)
{
  auto uMap = tt::buildMonolithicMapNFieldFromNodeMap(ownedNodeMap, NglobNodes, /*nFields=*/2);

  auto Auu = Teuchos::rcp(new crs_type(uMap, nnzPerRowUQ1Estimate()));
  Auu->resumeFill();

  auto GO_0 = GO(0);

  addShiftedBlockInto(*Auu, *Kxx, /*rowOff=*/GO_0,          /*colOff=*/GO_0);
  addShiftedBlockInto(*Auu, *Kxy, /*rowOff=*/GO_0,          /*colOff=*/NglobNodes);
  addShiftedBlockInto(*Auu, *Kyx, /*rowOff=*/NglobNodes, /*colOff=*/GO_0);
  addShiftedBlockInto(*Auu, *Kyy, /*rowOff=*/NglobNodes, /*colOff=*/NglobNodes);

  Auu->fillComplete(uMap, uMap);
  return Auu;
}

inline Teuchos::RCP<crs_type>
buildAuT_from_scalar_blocks(const Teuchos::RCP<const map_type>& ownedNodeMap,
                            GO NglobNodes,
                            const Teuchos::RCP<crs_type>& KxT,
                            const Teuchos::RCP<crs_type>& KyT)
{
  auto uMap = tt::buildMonolithicMapNFieldFromNodeMap(ownedNodeMap, NglobNodes, /*nFields=*/2);
  auto tMap = ownedNodeMap;

  auto AuT = Teuchos::rcp(new crs_type(uMap, nnzPerRowRectEstimate()));
  AuT->resumeFill();
  
  auto GO_0 = GO(0);

  addShiftedBlockInto(*AuT, *KxT, /*rowOff=*/GO_0,          /*colOff=*/GO_0);
  addShiftedBlockInto(*AuT, *KyT, /*rowOff=*/NglobNodes, /*colOff=*/GO_0);

  // domain = tMap, range = uMap
  AuT->fillComplete(tMap, uMap);
  return AuT;
}

inline Teuchos::RCP<crs_type>
buildATu_from_scalar_blocks(const Teuchos::RCP<const map_type>& ownedNodeMap,
                            GO NglobNodes,
                            const Teuchos::RCP<crs_type>& KTx,
                            const Teuchos::RCP<crs_type>& KTy)
{
  auto uMap = tt::buildMonolithicMapNFieldFromNodeMap(ownedNodeMap, NglobNodes, /*nFields=*/2);
  auto tMap = ownedNodeMap;

  auto ATu = Teuchos::rcp(new crs_type(tMap, nnzPerRowRectEstimate()));
  ATu->resumeFill();

  auto GO_0 = GO(0);

  addShiftedBlockInto(*ATu, *KTx, /*rowOff=*/GO_0, /*colOff=*/GO_0);
  addShiftedBlockInto(*ATu, *KTy, /*rowOff=*/GO_0, /*colOff=*/NglobNodes);

  // domain = uMap, range = tMap
  ATu->fillComplete(uMap, tMap);
  return ATu;
}

// ---------------- Element assembly ----------------
//
// Builds element matrices for:
//  - Elasticity stiffness blocks (Kxx,Kxy,Kyx,Kyy)
//  - Heat advection-diffusion (AT) (nonsymmetric if w != 0)
//  - Coupling blocks:
//     mechanics <- T : KxT, KyT  (from -∫ beta (div v) T )
//     heat <- u      : KTx, KTy  (from -∫ eta s div(u) )
inline void assemble_element_matrices_Q1(
    const SC x[4], const SC y[4],
    const Params& p,
    SC Kxx[4][4], SC Kxy[4][4], SC Kyx[4][4], SC Kyy[4][4],
    SC AT [4][4],
    SC KxT[4][4], SC KyT[4][4],
    SC KTx[4][4], SC KTy[4][4])
{
  for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
    Kxx[i][j]=Kxy[i][j]=Kyx[i][j]=Kyy[i][j]=0.0;
    AT [i][j]=0.0;
    KxT[i][j]=KyT[i][j]=0.0;
    KTx[i][j]=KTy[i][j]=0.0;
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

    // Elasticity bilinear form: ∫ (2μ ε(u):ε(v) + λ div(u) div(v))
    for (int i=0;i<4;++i) {
      for (int j=0;j<4;++j) {
        const SC dix = dN_dx[i], diy = dN_dy[i];
        const SC djx = dN_dx[j], djy = dN_dy[j];

        // vx with ux
        Kxx[i][j] += (2*p.mu * (dix*djx)
                      + 2*p.mu * (0.5*diy)*(0.5*djy)
                      + p.lambda*(dix)*(djx)) * dV;

        // vy with uy
        Kyy[i][j] += (2*p.mu * (diy*djy)
                      + 2*p.mu * (0.5*dix)*(0.5*djx)
                      + p.lambda*(diy)*(djy)) * dV;

        // vx with uy
        Kxy[i][j] += (2*p.mu * (0.5*diy)*(0.5*djx)
                      + p.lambda*(dix)*(djy)) * dV;

        // vy with ux
        Kyx[i][j] += (2*p.mu * (0.5*dix)*(0.5*djy)
                      + p.lambda*(diy)*(djx)) * dV;
      }
    }

    // mechanics <- T : -∫ beta (div v) T
    for (int i=0;i<4;++i) {
      for (int j=0;j<4;++j) {
        KxT[i][j] += (-p.beta) * (dN_dx[i]) * (N[j]) * dV;
        KyT[i][j] += (-p.beta) * (dN_dy[i]) * (N[j]) * dV;
      }
    }

    // heat: ∫ kappa ∇s·∇T + ∫ rhoCp s (w·∇T)
    for (int i=0;i<4;++i) {
      for (int j=0;j<4;++j) {
        AT[i][j] += p.kappa * (dN_dx[i]*dN_dx[j] + dN_dy[i]*dN_dy[j]) * dV;
        AT[i][j] += p.rhoCp * (N[i]) * (p.wx*dN_dx[j] + p.wy*dN_dy[j]) * dV; // nonsym
      }
    }

    // heat <- u : -∫ eta s div(u) = -∫ eta s (dux/dx + duy/dy)
    for (int i=0;i<4;++i) {
      for (int j=0;j<4;++j) {
        KTx[i][j] += (-p.eta) * (N[i]) * (dN_dx[j]) * dV;
        KTy[i][j] += (-p.eta) * (N[i]) * (dN_dy[j]) * dV;
      }
    }
  }
}

inline void assertColMapLooksLikeNodeGIDs(const crs_type& A, GO NglobNodes, const std::string& name)
{
  auto colMap = A.getColMap();
  const LO nLocalCols = (LO)colMap->getLocalNumElements();
  for (LO lc=0; lc<nLocalCols; ++lc) {
    const GO g = colMap->getGlobalElement(lc);
    if (g < 0 || g >= NglobNodes) {
      throw std::runtime_error(name + ": colMap GID out of node range [0,NglobNodes): " +
                               std::to_string((long long)g));
    }
  }
}

inline Teuchos::RCP<Tpetra::CrsGraph<LO,GO,NO>>
buildGraphAuu(const Teuchos::RCP<const map_type>& ownedNodeMap,
              GO NglobNodes,
              const Teuchos::RCP<const crs_type>& Kxx,
              const Teuchos::RCP<const crs_type>& Kxy,
              const Teuchos::RCP<const crs_type>& Kyx,
              const Teuchos::RCP<const crs_type>& Kyy,
              const Teuchos::RCP<const map_type>& uMap)
{
  using graph_type = Tpetra::CrsGraph<LO,GO,NO>;

  const LO nLocalNodes = (LO)ownedNodeMap->getLocalNumElements();
  const LO nLocalU     = 2 * nLocalNodes;

  Teuchos::ArrayRCP<size_t> nnzPerRow(nLocalU);

  // Compute exact nnz per u-row:
  // ux row i: nnz(Kxx row i) + nnz(Kxy row i)
  // uy row i: nnz(Kyx row i) + nnz(Kyy row i)
  for (LO i=0; i<nLocalNodes; ++i) {
    const GO nodeG = ownedNodeMap->getGlobalElement(i);
    const LO lr = Kxx->getRowMap()->getLocalElement(nodeG);

    nnzPerRow[i]            = Kxx->getNumEntriesInLocalRow(lr) + Kxy->getNumEntriesInLocalRow(lr);
    nnzPerRow[i+nLocalNodes]= Kyx->getNumEntriesInLocalRow(lr) + Kyy->getNumEntriesInLocalRow(lr);
  }

  auto G = Teuchos::rcp(new graph_type(uMap, nnzPerRow()));
  // Insert column GIDs
  for (LO i=0; i<nLocalNodes; ++i) {
    const GO nodeG = ownedNodeMap->getGlobalElement(i);
    const LO lr = Kxx->getRowMap()->getLocalElement(nodeG);

    // ux row
    {
      const GO rowG = nodeG;

      typename crs_type::local_inds_host_view_type inds;
      typename crs_type::values_host_view_type     vals;

      std::vector<GO> cols;
      cols.reserve(nnzPerRow[i]);

      // Kxx -> ux cols
      Kxx->getLocalRowView(lr, inds, vals);
      auto colMap = Kxx->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) cols.push_back(colMap->getGlobalElement(inds(k)));

      // Kxy -> uy cols shifted
      Kxy->getLocalRowView(lr, inds, vals);
      colMap = Kxy->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) cols.push_back(colMap->getGlobalElement(inds(k)) + NglobNodes);

      if (!cols.empty()) G->insertGlobalIndices(rowG, Teuchos::arrayView(cols.data(), cols.size()));
    }

    // uy row
    {
      const GO rowG = nodeG + NglobNodes;

      typename crs_type::local_inds_host_view_type inds;
      typename crs_type::values_host_view_type     vals;

      std::vector<GO> cols;
      cols.reserve(nnzPerRow[i+nLocalNodes]);

      // Kyx -> ux cols
      Kyx->getLocalRowView(lr, inds, vals);
      auto colMap = Kyx->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) cols.push_back(colMap->getGlobalElement(inds(k)));

      // Kyy -> uy cols shifted
      Kyy->getLocalRowView(lr, inds, vals);
      colMap = Kyy->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) cols.push_back(colMap->getGlobalElement(inds(k)) + NglobNodes);

      if (!cols.empty()) G->insertGlobalIndices(rowG, Teuchos::arrayView(cols.data(), cols.size()));
    }
  }

  G->fillComplete(uMap, uMap);
  return G;
}

inline Teuchos::RCP<Tpetra::CrsGraph<LO,GO,NO>>
buildGraphAuT(const Teuchos::RCP<const map_type>& ownedNodeMap,
              GO NglobNodes,
              const Teuchos::RCP<const crs_type>& KxT,
              const Teuchos::RCP<const crs_type>& KyT,
              const Teuchos::RCP<const map_type>& uMap,
              const Teuchos::RCP<const map_type>& tMap)
{
  using graph_type = Tpetra::CrsGraph<LO,GO,NO>;

  const LO nLocalNodes = (LO)ownedNodeMap->getLocalNumElements();
  const LO nLocalU     = 2 * nLocalNodes;

  Teuchos::ArrayRCP<size_t> nnzPerRow(nLocalU);

  for (LO i=0; i<nLocalNodes; ++i) {
    const GO nodeG = ownedNodeMap->getGlobalElement(i);
    const LO lr = KxT->getRowMap()->getLocalElement(nodeG);
    nnzPerRow[i]             = KxT->getNumEntriesInLocalRow(lr);
    nnzPerRow[i+nLocalNodes] = KyT->getNumEntriesInLocalRow(lr);
  }

  auto G = Teuchos::rcp(new graph_type(uMap, nnzPerRow()));

  for (LO i=0; i<nLocalNodes; ++i) {
    const GO nodeG = ownedNodeMap->getGlobalElement(i);
    const LO lr = KxT->getRowMap()->getLocalElement(nodeG);

    typename crs_type::local_inds_host_view_type inds;
    typename crs_type::values_host_view_type     vals;

    // ux row (nodeG)
    {
      const GO rowG = nodeG;
      KxT->getLocalRowView(lr, inds, vals);

      std::vector<GO> cols;
      cols.reserve(nnzPerRow[i]);
      auto colMap = KxT->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) cols.push_back(colMap->getGlobalElement(inds(k))); // T cols (no shift)

      if (!cols.empty()) G->insertGlobalIndices(rowG, Teuchos::arrayView(cols.data(), cols.size()));
    }

    // uy row (nodeG+N)
    {
      const GO rowG = nodeG + NglobNodes;
      KyT->getLocalRowView(lr, inds, vals);

      std::vector<GO> cols;
      cols.reserve(nnzPerRow[i+nLocalNodes]);
      auto colMap = KyT->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) cols.push_back(colMap->getGlobalElement(inds(k))); // T cols

      if (!cols.empty()) G->insertGlobalIndices(rowG, Teuchos::arrayView(cols.data(), cols.size()));
    }
  }

  G->fillComplete(tMap, uMap); // domain=tMap, range=uMap
  return G;
}

inline Teuchos::RCP<Tpetra::CrsGraph<LO,GO,NO>>
buildGraphATu(const Teuchos::RCP<const map_type>& ownedNodeMap,
              GO NglobNodes,
              const Teuchos::RCP<const crs_type>& KTx,
              const Teuchos::RCP<const crs_type>& KTy,
              const Teuchos::RCP<const map_type>& tMap,
              const Teuchos::RCP<const map_type>& uMap)
{
  using graph_type = Tpetra::CrsGraph<LO,GO,NO>;

  const LO nLocalNodes = (LO)ownedNodeMap->getLocalNumElements();
  Teuchos::ArrayRCP<size_t> nnzPerRow(nLocalNodes);

  for (LO i=0; i<nLocalNodes; ++i) {
    const GO nodeG = ownedNodeMap->getGlobalElement(i);
    const LO lr = KTx->getRowMap()->getLocalElement(nodeG);
    nnzPerRow[i] = KTx->getNumEntriesInLocalRow(lr) + KTy->getNumEntriesInLocalRow(lr);
  }

  auto G = Teuchos::rcp(new graph_type(tMap, nnzPerRow()));

  for (LO i=0; i<nLocalNodes; ++i) {
    const GO nodeG = ownedNodeMap->getGlobalElement(i);
    const LO lr = KTx->getRowMap()->getLocalElement(nodeG);

    typename crs_type::local_inds_host_view_type inds;
    typename crs_type::values_host_view_type     vals;

    std::vector<GO> cols;
    cols.reserve(nnzPerRow[i]);

    // KTx -> ux cols
    KTx->getLocalRowView(lr, inds, vals);
    auto colMap = KTx->getColMap();
    for (size_t k=0;k<inds.extent(0);++k) cols.push_back(colMap->getGlobalElement(inds(k)));

    // KTy -> uy cols shifted
    KTy->getLocalRowView(lr, inds, vals);
    colMap = KTy->getColMap();
    for (size_t k=0;k<inds.extent(0);++k) cols.push_back(colMap->getGlobalElement(inds(k)) + NglobNodes);

    if (!cols.empty()) G->insertGlobalIndices(nodeG, Teuchos::arrayView(cols.data(), cols.size()));
  }

  G->fillComplete(uMap, tMap); // domain=uMap, range=tMap
  return G;
}

inline Teuchos::RCP<crs_type>
buildAuu_fromGraphAndBlocks(const Teuchos::RCP<const map_type>& ownedNodeMap,
                            GO NglobNodes,
                            const Teuchos::RCP<const crs_type>& Kxx,
                            const Teuchos::RCP<const crs_type>& Kxy,
                            const Teuchos::RCP<const crs_type>& Kyx,
                            const Teuchos::RCP<const crs_type>& Kyy)
{
  auto uMap = tt::buildMonolithicMapNFieldFromNodeMap(ownedNodeMap, NglobNodes, 2);
  auto G = buildGraphAuu(ownedNodeMap, NglobNodes, Kxx, Kxy, Kyx, Kyy, uMap);

  auto Auu = Teuchos::rcp(new crs_type(G));
  Auu->resumeFill();

  const LO nLocalNodes = (LO)ownedNodeMap->getLocalNumElements();

  for (LO lnode=0; lnode<nLocalNodes; ++lnode) {
    const GO nodeG = ownedNodeMap->getGlobalElement(lnode);

    // ux row
    {
      const GO rowG = nodeG;
      const LO lrow = Kxx->getRowMap()->getLocalElement(nodeG);

      typename crs_type::local_inds_host_view_type inds;
      typename crs_type::values_host_view_type     v;

      // Kxx values (ux cols)
      Kxx->getLocalRowView(lrow, inds, v);
      std::vector<GO> cols(inds.extent(0));
      std::vector<SC> vals(inds.extent(0));
      auto colMap = Kxx->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) { cols[k]=colMap->getGlobalElement(inds(k)); vals[k]=v(k); }
      if (!cols.empty()) Auu->sumIntoGlobalValues(rowG, (LO)cols.size(), vals.data(), cols.data());

      // Kxy values (uy cols shifted)
      Kxy->getLocalRowView(lrow, inds, v);
      cols.resize(inds.extent(0));
      vals.resize(inds.extent(0));
      colMap = Kxy->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) { cols[k]=colMap->getGlobalElement(inds(k)) + NglobNodes; vals[k]=v(k); }
      if (!cols.empty()) Auu->sumIntoGlobalValues(rowG, (LO)cols.size(), vals.data(), cols.data());
    }

    // uy row
    {
      const GO rowG = nodeG + NglobNodes;
      const LO lrow = Kyy->getRowMap()->getLocalElement(nodeG);

      typename crs_type::local_inds_host_view_type inds;
      typename crs_type::values_host_view_type     v;

      // Kyx values (ux cols)
      Kyx->getLocalRowView(lrow, inds, v);
      std::vector<GO> cols(inds.extent(0));
      std::vector<SC> vals(inds.extent(0));
      auto colMap = Kyx->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) { cols[k]=colMap->getGlobalElement(inds(k)); vals[k]=v(k); }
      if (!cols.empty()) Auu->sumIntoGlobalValues(rowG, (LO)cols.size(), vals.data(), cols.data());

      // Kyy values (uy cols shifted)
      Kyy->getLocalRowView(lrow, inds, v);
      cols.resize(inds.extent(0));
      vals.resize(inds.extent(0));
      colMap = Kyy->getColMap();
      for (size_t k=0;k<inds.extent(0);++k) { cols[k]=colMap->getGlobalElement(inds(k)) + NglobNodes; vals[k]=v(k); }
      if (!cols.empty()) Auu->sumIntoGlobalValues(rowG, (LO)cols.size(), vals.data(), cols.data());
    }
  }

  Auu->fillComplete(uMap, uMap);
  return Auu;
}

inline Teuchos::RCP<crs_type>
buildAuT_fromGraphAndBlocks(const Teuchos::RCP<const map_type>& ownedNodeMap,
                            GO NglobNodes,
                            const Teuchos::RCP<const crs_type>& KxT,
                            const Teuchos::RCP<const crs_type>& KyT)
{
  auto uMap = tt::buildMonolithicMapNFieldFromNodeMap(ownedNodeMap, NglobNodes, /*numFields=*/2);
  auto tMap = ownedNodeMap;

  // Build static graph
  auto G = buildGraphAuT(ownedNodeMap, NglobNodes, KxT, KyT, uMap, tMap);

  // Create matrix with that graph
  auto AuT = Teuchos::rcp(new crs_type(G));
  AuT->resumeFill();

  const LO nLocalNodes = (LO)ownedNodeMap->getLocalNumElements();

  for (LO i=0; i<nLocalNodes; ++i) {
    const GO nodeG = ownedNodeMap->getGlobalElement(i);
    const LO lr = KxT->getRowMap()->getLocalElement(nodeG);

    typename crs_type::local_inds_host_view_type inds;
    typename crs_type::values_host_view_type     v;

    // ux row (nodeG): from KxT
    {
      const GO rowG = nodeG;
      KxT->getLocalRowView(lr, inds, v);

      const size_t n = inds.extent(0);
      if (n > 0) {
        std::vector<GO> cols(n);
        std::vector<SC> vals(n);
        auto colMap = KxT->getColMap();
        for (size_t k=0;k<n;++k) {
          cols[k] = colMap->getGlobalElement(inds(k)); // T columns (no shift)
          vals[k] = v(k);
        }
        AuT->sumIntoGlobalValues(rowG, (LO)n, vals.data(), cols.data());
      }
    }

    // uy row (nodeG + NglobNodes): from KyT
    {
      const GO rowG = nodeG + NglobNodes;
      KyT->getLocalRowView(lr, inds, v);

      const size_t n = inds.extent(0);
      if (n > 0) {
        std::vector<GO> cols(n);
        std::vector<SC> vals(n);
        auto colMap = KyT->getColMap();
        for (size_t k=0;k<n;++k) {
          cols[k] = colMap->getGlobalElement(inds(k)); // T columns (no shift)
          vals[k] = v(k);
        }
        AuT->sumIntoGlobalValues(rowG, (LO)n, vals.data(), cols.data());
      }
    }
  }

  // domain=tMap (T), range=uMap (u)
  AuT->fillComplete(tMap, uMap);
  return AuT;
}

inline Teuchos::RCP<crs_type>
buildATu_fromGraphAndBlocks(const Teuchos::RCP<const map_type>& ownedNodeMap,
                            GO NglobNodes,
                            const Teuchos::RCP<const crs_type>& KTx,
                            const Teuchos::RCP<const crs_type>& KTy)
{
  auto uMap = tt::buildMonolithicMapNFieldFromNodeMap(ownedNodeMap, NglobNodes, /*numFields=*/2);
  auto tMap = ownedNodeMap;

  // Build static graph
  auto G = buildGraphATu(ownedNodeMap, NglobNodes, KTx, KTy, tMap, uMap);

  // Create matrix with that graph
  auto ATu = Teuchos::rcp(new crs_type(G));
  ATu->resumeFill();

  const LO nLocalNodes = (LO)ownedNodeMap->getLocalNumElements();

  for (LO i=0; i<nLocalNodes; ++i) {
    const GO nodeG = ownedNodeMap->getGlobalElement(i);
    const LO lr = KTx->getRowMap()->getLocalElement(nodeG);

    typename crs_type::local_inds_host_view_type inds;
    typename crs_type::values_host_view_type     v;

    // Build one T row (nodeG) containing ux columns (from KTx) and uy columns shifted (from KTy)
    const GO rowG = nodeG;

    // We'll accumulate into vectors and do one sumInto call (not required, but tidy)
    std::vector<GO> cols;
    std::vector<SC> vals;

    // KTx: ux columns
    KTx->getLocalRowView(lr, inds, v);
    {
      const size_t n = inds.extent(0);
      cols.reserve(cols.size() + n);
      vals.reserve(vals.size() + n);
      auto colMap = KTx->getColMap();
      for (size_t k=0;k<n;++k) {
        cols.push_back(colMap->getGlobalElement(inds(k))); // ux col
        vals.push_back(v(k));
      }
    }

    // KTy: uy columns shifted
    KTy->getLocalRowView(lr, inds, v);
    {
      const size_t n = inds.extent(0);
      cols.reserve(cols.size() + n);
      vals.reserve(vals.size() + n);
      auto colMap = KTy->getColMap();
      for (size_t k=0;k<n;++k) {
        cols.push_back(colMap->getGlobalElement(inds(k)) + NglobNodes); // uy col
        vals.push_back(v(k));
      }
    }

    if (!cols.empty())
      ATu->sumIntoGlobalValues(rowG, (LO)cols.size(), vals.data(), cols.data());
  }

  // domain=uMap (u), range=tMap (T)
  ATu->fillComplete(uMap, tMap);
  return ATu;
}

// Apply homogeneous Dirichlet (ux=uy=T=0) to scalar nodal subblocks (all NxN).
// boundaryNodeGIDs: global node IDs on the boundary, in [0, NglobNodes).
template<class SC_, class LO_, class GO_, class NO_>
void applyHomogeneousDirichlet_ThermoelasticScalarBlocks(
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& Kxx,
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& Kxy,
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& Kyx,
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& Kyy,
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& AT,
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& KxT,
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& KyT,
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& KTx,
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& KTy,
    Teuchos::ArrayView<GO_> boundaryNodeGIDs)
{
  // Mechanics rows: ux, uy
  tt::applyDirichletRows_DiagBlock(Kxx, boundaryNodeGIDs);
  tt::applyDirichletRows_DiagBlock(Kyy, boundaryNodeGIDs);

  tt::applyDirichletRows_OffDiagBlock(Kxy, boundaryNodeGIDs);
  tt::applyDirichletRows_OffDiagBlock(Kyx, boundaryNodeGIDs);

  tt::applyDirichletRows_OffDiagBlock(KxT, boundaryNodeGIDs);
  tt::applyDirichletRows_OffDiagBlock(KyT, boundaryNodeGIDs);

  // Heat rows: T
  tt::applyDirichletRows_DiagBlock(AT, boundaryNodeGIDs);

  tt::applyDirichletRows_OffDiagBlock(KTx, boundaryNodeGIDs);
  tt::applyDirichletRows_OffDiagBlock(KTy, boundaryNodeGIDs);
}

// ---------------- Main builder: overlap assembly + export to owned + explicit Auu/AuT/ATu ----------------
inline BlocksOwned buildThermoelasticBlocks_OverlapExported(
    int Nx, int Ny, SC x0, SC x1, SC y0, SC y1,
    const Params& p)
{
  auto comm = Tpetra::getDefaultComm();
  const GO N = GO(Nx)*GO(Ny);
  auto ownedMap = Teuchos::rcp(new map_type(N, 0, comm));

  ConnView ownedElemConn = tt::buildOwnedElementConnectivity(ownedMap, Nx, Ny);
  auto overlapMap = tt::buildOverlapNodeMap(ownedMap, ownedElemConn);
  CoordView coords = tt::buildCoordsStructured(overlapMap, Nx, Ny, x0,x1,y0,y1);

  auto G_ov = buildNodeGraph(overlapMap, ownedElemConn);

  // overlap matrices (all scalar nodal)
  auto Kxx_ov = Teuchos::rcp(new crs_type(G_ov));
  auto Kxy_ov = Teuchos::rcp(new crs_type(G_ov));
  auto Kyx_ov = Teuchos::rcp(new crs_type(G_ov));
  auto Kyy_ov = Teuchos::rcp(new crs_type(G_ov));

  auto AT_ov  = Teuchos::rcp(new crs_type(G_ov));

  auto KxT_ov = Teuchos::rcp(new crs_type(G_ov));
  auto KyT_ov = Teuchos::rcp(new crs_type(G_ov));
  auto KTx_ov = Teuchos::rcp(new crs_type(G_ov));
  auto KTy_ov = Teuchos::rcp(new crs_type(G_ov));

  // element loop (owned elements) assemble into overlap matrices
  for (size_t e=0; e<ownedElemConn.extent(0); ++e) {
    GO gid[4]; LO lid[4];
    for (int a=0;a<4;++a) {
      gid[a] = ownedElemConn(e,a);
      lid[a] = overlapMap->getLocalElement(gid[a]);
      if (lid[a] == Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("Thermoelastic: element node missing from overlap map");
    }

    SC x[4], y[4];
    for (int a=0;a<4;++a) { x[a]=coords(lid[a],0); y[a]=coords(lid[a],1); }

    SC Kxx[4][4], Kxy[4][4], Kyx[4][4], Kyy[4][4], AT[4][4], KxT[4][4], KyT[4][4], KTx[4][4], KTy[4][4];
    assemble_element_matrices_Q1(x,y,p, Kxx,Kxy,Kyx,Kyy, AT, KxT,KyT, KTx,KTy);

    Teuchos::Array<LO> lcols(4);
    Teuchos::Array<SC> vals(4);

    for (int i=0;i<4;++i) {
      const LO lrow = lid[i];
      for (int j=0;j<4;++j) lcols[j]=lid[j];

      for (int j=0;j<4;++j) vals[j]=Kxx[i][j];
      Kxx_ov->sumIntoLocalValues(lrow, lcols(), vals());

      for (int j=0;j<4;++j) vals[j]=Kxy[i][j];
      Kxy_ov->sumIntoLocalValues(lrow, lcols(), vals());

      for (int j=0;j<4;++j) vals[j]=Kyx[i][j];
      Kyx_ov->sumIntoLocalValues(lrow, lcols(), vals());

      for (int j=0;j<4;++j) vals[j]=Kyy[i][j];
      Kyy_ov->sumIntoLocalValues(lrow, lcols(), vals());

      for (int j=0;j<4;++j) vals[j]=AT[i][j];
      AT_ov->sumIntoLocalValues(lrow, lcols(), vals());

      for (int j=0;j<4;++j) vals[j]=KxT[i][j];
      KxT_ov->sumIntoLocalValues(lrow, lcols(), vals());

      for (int j=0;j<4;++j) vals[j]=KyT[i][j];
      KyT_ov->sumIntoLocalValues(lrow, lcols(), vals());

      for (int j=0;j<4;++j) vals[j]=KTx[i][j];
      KTx_ov->sumIntoLocalValues(lrow, lcols(), vals());

      for (int j=0;j<4;++j) vals[j]=KTy[i][j];
      KTy_ov->sumIntoLocalValues(lrow, lcols(), vals());
    }
  }

  // fillComplete overlap
  Kxx_ov->fillComplete(overlapMap, overlapMap);
  Kxy_ov->fillComplete(overlapMap, overlapMap);
  Kyx_ov->fillComplete(overlapMap, overlapMap);
  Kyy_ov->fillComplete(overlapMap, overlapMap);

  AT_ov ->fillComplete(overlapMap, overlapMap);

  KxT_ov->fillComplete(overlapMap, overlapMap);
  KyT_ov->fillComplete(overlapMap, overlapMap);
  KTx_ov->fillComplete(overlapMap, overlapMap);
  KTy_ov->fillComplete(overlapMap, overlapMap);

  // export to owned and build explicit monolithic blocks
  BlocksOwned out;
  out.Kxx = exportToOwned(Kxx_ov, ownedMap, overlapMap);
  out.Kxy = exportToOwned(Kxy_ov, ownedMap, overlapMap);
  out.Kyx = exportToOwned(Kyx_ov, ownedMap, overlapMap);
  out.Kyy = exportToOwned(Kyy_ov, ownedMap, overlapMap);

  out.AT  = exportToOwned(AT_ov,  ownedMap, overlapMap);

  out.KxT = exportToOwned(KxT_ov, ownedMap, overlapMap);
  out.KyT = exportToOwned(KyT_ov, ownedMap, overlapMap);
  out.KTx = exportToOwned(KTx_ov, ownedMap, overlapMap);
  out.KTy = exportToOwned(KTy_ov, ownedMap, overlapMap);

  auto boundary = tt::boundaryNodeGIDs(Nx, Ny);

  applyHomogeneousDirichlet_ThermoelasticScalarBlocks(
      *out.Kxx, *out.Kxy, *out.Kyx, *out.Kyy,
      *out.AT,
      *out.KxT, *out.KyT,
      *out.KTx, *out.KTy,
      boundary());

  const GO NglobNodes = GO(Nx) * GO(Ny);

  out.Auu = buildAuu_fromGraphAndBlocks(ownedMap, NglobNodes, out.Kxx, out.Kxy, out.Kyx, out.Kyy);
  out.AuT = buildAuT_fromGraphAndBlocks(ownedMap, NglobNodes, out.KxT, out.KyT);
  out.ATu = buildATu_fromGraphAndBlocks(ownedMap, NglobNodes, out.KTx, out.KTy);

  return out;
}

} // namespace tt::thermoelastic