/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

// main_pnp.cpp (linear-drift Poisson + Nernst–Planck MMS benchmark, 2-field blocked/monolithic)
//
// Unknown ordering (monolithic 2-field, field-blocked by node offset):
//   phi(nodeG)         -> dofGID = nodeGID
//   c(nodeG)           -> dofGID = nodeGID + NglobNodes
//
// PDEs (steady, linear coupled):
//   -div(eps grad phi) = rho_f + zF c
//    div(-D grad c - alpha grad phi + u c) = s
//
// Dirichlet BCs on both fields using MMS exact boundary values.
// Additionally: apply the same Dirichlet row treatment at the *subblock level*
// (A00,A01,A10,A11 and RHS block entries) so block preconditioners do not see a nullspace.
//
// This file adds 2x2 blocked operator paths (teko-bgs, teko-amg) exactly like thermoelastic.
//
// Requires:
//   multiphys_mms_pnp_linear.hpp
//   multiphys_pnp_linear_q1.hpp   (for assembly routines; we only use its element assembly logic indirectly here)
// and your existing framework headers:
//   multiphys_mesh.hpp, multiphys_q1_assembly.hpp, multiphys_dof.hpp, multiphys_dirichlet_blocks.hpp, multiphys_dirichlet.hpp,
//   multiphys_transfer.hpp, multiphys_solvers.hpp

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>

#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>

#include <BelosTpetraAdapter.hpp>

#include "Teko_Utilities.hpp"

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <cmath>
#include <vector>
#include <stdexcept>

#include "multiphys_mesh.hpp"
#include "multiphys_q1_assembly.hpp"
#include "multiphys_transfer.hpp"
#include "multiphys_solvers.hpp"

#include "multiphys_dof.hpp"
#include "multiphys_dirichlet.hpp"
#include "multiphys_dirichlet_blocks.hpp"

#include "multiphys_mms_pnp_linear.hpp"

namespace {

using SC = tt::SC;
using LO = tt::LO;
using GO = tt::GO;
using NO = tt::NO;

using crs_type = tt::crs_type;
using mv_type  = Tpetra::MultiVector<SC,LO,GO,NO>;
using vec_type = tt::vec_type;

template<class SC_, class LO_, class GO_, class NO_>
Teuchos::Array<typename Teuchos::ScalarTraits<SC_>::magnitudeType>
explicitResidualNorm2(const Tpetra::Operator<SC_,LO_,GO_,NO_>& A,
                      const Tpetra::MultiVector<SC_,LO_,GO_,NO_>& x,
                      const Tpetra::MultiVector<SC_,LO_,GO_,NO_>& b)
{
  using MV = Tpetra::MultiVector<SC_,LO_,GO_,NO_>;
  MV r(b.getMap(), b.getNumVectors());
  A.apply(x, r);
  r.update(1.0, b, -1.0);
  Teuchos::Array<typename Teuchos::ScalarTraits<SC_>::magnitudeType> n(b.getNumVectors());
  r.norm2(n());
  return n;
}

template<class SC_, class LO_, class GO_, class NO_>
struct L2Error2Field {
  typename Teuchos::ScalarTraits<SC_>::magnitudeType ephi;
  typename Teuchos::ScalarTraits<SC_>::magnitudeType ec;
  typename Teuchos::ScalarTraits<SC_>::magnitudeType combined;
};

template<class SC_, class LO_, class GO_, class NO_>
L2Error2Field<SC_,LO_,GO_,NO_>
discreteL2ErrorMonolithic2Field(const Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& M,
                                const Tpetra::MultiVector<SC_,LO_,GO_,NO_>& x,
                                const Tpetra::MultiVector<SC_,LO_,GO_,NO_>& xe)
{
  using MV = Tpetra::MultiVector<SC_,LO_,GO_,NO_>;
  using mag_type = typename Teuchos::ScalarTraits<SC_>::magnitudeType;

  const LO_ nLocalMono = (LO_)x.getLocalLength();
  TEUCHOS_TEST_FOR_EXCEPTION(nLocalMono % 2 != 0, std::runtime_error,
                             "discreteL2ErrorMonolithic2Field: local length not divisible by 2");
  const LO_ nLocalNode = nLocalMono / 2;

  MV ephi(M.getRowMap(), 1), ec(M.getRowMap(), 1);
  {
    auto xv  = x.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto xev = xe.getLocalViewHost(Tpetra::Access::ReadOnly);

    auto ephi_v = ephi.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto ec_v   = ec.getLocalViewHost(Tpetra::Access::OverwriteAll);

    for (LO_ i=0;i<nLocalNode;++i) {
      ephi_v(i,0) = xv(i,0)            - xev(i,0);
      ec_v(i,0)   = xv(i+nLocalNode,0) - xev(i+nLocalNode,0);
    }
  }

  MV Mephi(M.getRowMap(), 1), Mec(M.getRowMap(), 1);
  M.apply(ephi, Mephi);
  M.apply(ec,   Mec);

  Teuchos::Array<mag_type> tmp(1);
  ephi.dot(Mephi, tmp()); mag_type ephi2 = tmp[0];
  ec.dot(Mec, tmp());     mag_type ec2   = tmp[0];

  ephi2 = (ephi2 < mag_type(0) ? mag_type(0) : ephi2);
  ec2   = (ec2   < mag_type(0) ? mag_type(0) : ec2);

  return {std::sqrt(ephi2), std::sqrt(ec2), std::sqrt(ephi2+ec2)};
}

// Assemble Q1 scalar blocks for linear-drift PNP on overlap map.
struct BlocksOverlap {
  Teuchos::RCP<crs_type> A00_phi_phi; // Poisson stiffness
  Teuchos::RCP<crs_type> A01_phi_c;   // -zF * M
  Teuchos::RCP<crs_type> A10_c_phi;   // alpha * stiffness
  Teuchos::RCP<crs_type> A11_c_c;     // D * stiffness + advection term
  Teuchos::RCP<mv_type>  bphi;
  Teuchos::RCP<mv_type>  bc;
  Teuchos::RCP<mv_type>  phi_exact;
  Teuchos::RCP<mv_type>  c_exact;
};

inline BlocksOverlap assembleBlocksOverlap_Q1_MMS(
    const Teuchos::RCP<const tt::map_type>& overlapNodeMap,
    const tt::CoordView& coordsOverlap,
    const tt::ConnView& ownedElemConn,
    int Nx, int Ny,
    SC x0, SC x1, SC y0, SC y1,
    SC eps, SC D, SC alpha, SC zF)
{
  (void)Ny;

  auto G_ov = tt::buildQ1GraphFromOwnedElements(overlapNodeMap, ownedElemConn);

  auto A00 = Teuchos::rcp(new crs_type(G_ov));
  auto A01 = Teuchos::rcp(new crs_type(G_ov));
  auto A10 = Teuchos::rcp(new crs_type(G_ov));
  auto A11 = Teuchos::rcp(new crs_type(G_ov));

  auto bphi = Teuchos::make_rcp<mv_type>(overlapNodeMap, 1);
  auto bc   = Teuchos::make_rcp<mv_type>(overlapNodeMap, 1);
  bphi->putScalar(0.0);
  bc->putScalar(0.0);

  auto phi_ex = Teuchos::make_rcp<mv_type>(overlapNodeMap, 1);
  auto c_ex   = Teuchos::make_rcp<mv_type>(overlapNodeMap, 1);
  phi_ex->putScalar(0.0);
  c_ex->putScalar(0.0);

  // Fill nodal exact (for later export / BC values check)
  {
    auto ph = phi_ex->getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto ch = c_ex->getLocalViewHost(Tpetra::Access::OverwriteAll);

    const LO nLocal = (LO)overlapNodeMap->getLocalNumElements();
    for (LO l=0; l<nLocal; ++l) {
      const SC X = coordsOverlap(l,0);
      const SC Y = coordsOverlap(l,1);
      ph(l,0) = tt::mms_pnp_linear::phi_ex(X,Y);
      ch(l,0) = tt::mms_pnp_linear::c_ex(X,Y);
    }
  }

  // Element quadrature
  const SC a = 1.0/std::sqrt(3.0);
  const SC gp[2] = {-a, a};

  for (size_t e=0; e<ownedElemConn.extent(0); ++e) {
    GO gid[4]; LO lid[4];
    for (int aN=0;aN<4;++aN) {
      gid[aN] = ownedElemConn(e,aN);
      lid[aN] = overlapNodeMap->getLocalElement(gid[aN]);
      if (lid[aN] == Teuchos::OrdinalTraits<LO>::invalid())
        throw std::runtime_error("PNP: element node missing from overlap map");
    }

    SC x[4], y[4];
    for (int aN=0;aN<4;++aN) { x[aN]=coordsOverlap(lid[aN],0); y[aN]=coordsOverlap(lid[aN],1); }

    // element matrices
    SC Ke[4][4];   // stiffness
    SC Me[4][4];   // mass
    SC Aadv[4][4]; // convection-like term: ∫ (-u N_j)·∇w_i
    SC fe[4];      // Poisson RHS load
    SC fc[4];      // NP RHS load

    for (int i=0;i<4;++i) {
      fe[i]=0.0; fc[i]=0.0;
      for (int j=0;j<4;++j) { Ke[i][j]=0.0; Me[i][j]=0.0; Aadv[i][j]=0.0; }
    }

    for (int ixi=0; ixi<2; ++ixi) for (int ieta=0; ieta<2; ++ieta) {
      const SC xi=gp[ixi], eta=gp[ieta];
      const SC wq = 1.0;

      SC N[4], dN_dxi[4], dN_deta[4];
      tt::shapeQ1(xi,eta,N,dN_dxi,dN_deta);

      SC xq=0.0, yq=0.0;
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

      SC dN_dx[4], dN_dy[4];
      for (int aN=0;aN<4;++aN) {
        dN_dx[aN] = invJ[0][0]*dN_dxi[aN] + invJ[1][0]*dN_deta[aN];
        dN_dy[aN] = invJ[0][1]*dN_dxi[aN] + invJ[1][1]*dN_deta[aN];
      }

      // MMS sources evaluated at quadrature
      const SC rhoF = tt::mms_pnp_linear::rho_f(xq,yq, eps, zF);
      const SC sNP  = tt::mms_pnp_linear::np_source(xq,yq, D, alpha);

      // u at quadrature (prescribed field)
      const SC ux = tt::mms_pnp_linear::ux(xq,yq);
      const SC uy = tt::mms_pnp_linear::uy(xq,yq);

      for (int i=0;i<4;++i) {
        fe[i] += rhoF * N[i] * dV;
        fc[i] += sNP  * N[i] * dV;

        for (int j=0;j<4;++j) {
          Ke[i][j] += (dN_dx[i]*dN_dx[j] + dN_dy[i]*dN_dy[j]) * dV;
          Me[i][j] += (N[i]*N[j]) * dV;
          Aadv[i][j] += (-(ux*dN_dx[i] + uy*dN_dy[i]) * N[j]) * dV; // -(u N_j)·∇w_i
        }
      }
    }

    // scatter element contributions to overlap matrices
    Teuchos::Array<LO> lcols(4);
    Teuchos::Array<SC> vals(4);

    for (int i=0;i<4;++i) {
      const LO lrow = lid[i];
      for (int j=0;j<4;++j) lcols[j]=lid[j];

      // A00 = eps * K
      for (int j=0;j<4;++j) vals[j] = eps * Ke[i][j];
      A00->sumIntoLocalValues(lrow, lcols(), vals());

      // A01 = -zF * M
      for (int j=0;j<4;++j) vals[j] = (-zF) * Me[i][j];
      A01->sumIntoLocalValues(lrow, lcols(), vals());

      // A10 = alpha * K
      for (int j=0;j<4;++j) vals[j] = alpha * Ke[i][j];
      A10->sumIntoLocalValues(lrow, lcols(), vals());

      // A11 = D*K + Aadv
      for (int j=0;j<4;++j) vals[j] = D * Ke[i][j] + Aadv[i][j];
      A11->sumIntoLocalValues(lrow, lcols(), vals());

      // RHS
      bphi->sumIntoLocalValue(lrow, 0, fe[i]);
      bc  ->sumIntoLocalValue(lrow, 0, fc[i]);
    }
  }

  // fillComplete overlap
  A00->fillComplete(overlapNodeMap, overlapNodeMap);
  A01->fillComplete(overlapNodeMap, overlapNodeMap);
  A10->fillComplete(overlapNodeMap, overlapNodeMap);
  A11->fillComplete(overlapNodeMap, overlapNodeMap);

  return {A00,A01,A10,A11,bphi,bc,phi_ex,c_ex};
}

// Apply Dirichlet at the scalar-block level so the blocked preconditioner sees nonsingular blocks.
//
// Strategy (simple and robust for Dirichlet on both fields):
// - For boundary rows in phi equation:
//     A00 row -> identity row
//     A01 row -> zero row
//     bphi(row) = phi_D
// - For boundary rows in c equation:
//     A11 row -> identity row
//     A10 row -> zero row
//     bc(row)  = c_D
//
// This matches the monolithic row replacement and avoids nullspace in diagonal blocks.
inline void applyDirichletToScalarBlocksAndRHS(
    crs_type& A00, crs_type& A01, crs_type& A10, crs_type& A11,
    mv_type& bphi, mv_type& bc,
    int Nx, int Ny,
    SC x0, SC x1, SC y0, SC y1)
{
  auto boundary = tt::boundaryNodeGIDs(Nx, Ny);

  // Replace rows in diagonal blocks / zero rows in off-diagonal blocks
  tt::applyDirichletRows_DiagBlock(A00, boundary());
  tt::applyDirichletRows_OffDiagBlock(A01, boundary());

  tt::applyDirichletRows_DiagBlock(A11, boundary());
  tt::applyDirichletRows_OffDiagBlock(A10, boundary());

  // Set RHS to Dirichlet values at boundary nodes (owned rows only)
  auto map = bphi.getMap();
  auto bph = bphi.getLocalViewHost(Tpetra::Access::ReadWrite);
  auto bch = bc.getLocalViewHost(Tpetra::Access::ReadWrite);

  const LO nLocal = (LO)map->getLocalNumElements();
  for (LO l=0; l<nLocal; ++l) {
    const GO nodeG = map->getGlobalElement(l);
    if (!tt::isBoundaryNode(nodeG, Nx, Ny)) continue;

    // recover coords for structured domain
    int i,j; tt::nodeIJ(nodeG, Nx, i, j);
    const SC X = x0 + (x1-x0)*SC(i)/SC(Nx-1);
    const SC Y = y0 + (y1-y0)*SC(j)/SC(Ny-1);

    bph(l,0) = tt::mms_pnp_linear::phi_ex(X,Y);
    bch(l,0) = tt::mms_pnp_linear::c_ex(X,Y);
  }
}

// Build monolithic vectors xexact = [phi_ex; c_ex] and b=[bphi; bc] on the owned monolithic map.
inline void packMonolithic2Field(const tt::map_type& ownedNodeMap,
                                 const tt::map_type& ownedMonoMap,
                                 GO NglobNodes,
                                 const mv_type& phiNode,
                                 const mv_type& cNode,
                                 mv_type& xmono)
{
  xmono.putScalar(0.0);
  auto X = xmono.getLocalViewHost(Tpetra::Access::OverwriteAll);
  auto ph = phiNode.getLocalViewHost(Tpetra::Access::ReadOnly);
  auto ch = cNode.getLocalViewHost(Tpetra::Access::ReadOnly);

  const LO nLocalNodes = (LO)ownedNodeMap.getLocalNumElements();
  for (LO l=0; l<nLocalNodes; ++l) {
    const GO nodeG = ownedNodeMap.getGlobalElement(l);
    const LO lphi = ownedMonoMap.getLocalElement(nodeG);
    const LO lc   = ownedMonoMap.getLocalElement(nodeG + NglobNodes);
    TEUCHOS_TEST_FOR_EXCEPTION(lphi==Teuchos::OrdinalTraits<LO>::invalid() ||
                               lc==Teuchos::OrdinalTraits<LO>::invalid(),
                               std::runtime_error, "packMonolithic2Field: missing dof in mono map");
    X(lphi,0) = ph(l,0);
    X(lc,0)   = ch(l,0);
  }
}

template<typename Comm>
void print_Teuchos_stacked_timer(const Teuchos::RCP<const Comm> & comm)
{
  auto stacked_timer = Teuchos::TimeMonitor::getStackedTimer();

  stacked_timer->stopBaseTimer();

  if(comm->getRank() == 0)
    std::cout << "Linear Solver Timing Summary:\n";

  comm->barrier();

  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = options.output_minmax = true;
  options.output_histogram = false;
  options.print_warnings = false;

  stacked_timer->report(std::cout, comm, options);

  comm->barrier();

  if(comm->getRank() == 0)
    std::cout << "\n";

  stacked_timer->startBaseTimer();
}

} // namespace

int main(int argc, char** argv)
{
  Tpetra::ScopeGuard scope(&argc, &argv);
  using namespace tt;

  auto comm = Tpetra::getDefaultComm();

  // ---- mesh defaults ----
  int Nx = 81;
  int Ny = 81;
  const SC x0=0.0, x1=1.0, y0=0.0, y1=1.0;

  // ---- physics defaults ----
  double eps   = 1.0;
  double D     = 1.0;
  double alpha = 1.0;
  double zF    = 1.0;

  // ---- solver defaults ----
  std::string solverName = "ifpack2";
  std::string solverXml  = "";
  int maxIters = 200;
  double tol = 1e-10;

  // ---- parse cmdline ----
  Teuchos::CommandLineProcessor clp(false, true);

  clp.setOption("Nx", &Nx, "Number of nodes in x direction (structured grid)");
  clp.setOption("Ny", &Ny, "Number of nodes in y direction (structured grid)");

  clp.setOption("eps",   &eps,   "Permittivity epsilon");
  clp.setOption("D",     &D,     "Diffusivity D");
  clp.setOption("alpha", &alpha, "Linear drift strength alpha");
  clp.setOption("zF",    &zF,    "Poisson coupling z*F");

  clp.setOption("solver", &solverName, "Solver: ifpack2 | teko-bgs | teko-amg");
  clp.setOption("solver-xml", &solverXml,
                "Optional XML file overriding the chosen solver parameter list");
  clp.setOption("max-iters", &maxIters, "GMRES maximum iterations");
  clp.setOption("tol", &tol, "GMRES convergence tolerance");

  auto rc = clp.parse(argc, argv);
  if (rc != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    return (rc == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED ? 0 : 1);

  TEUCHOS_TEST_FOR_EXCEPTION(Nx < 2 || Ny < 2, std::runtime_error, "Nx and Ny must be >= 2");
  TEUCHOS_TEST_FOR_EXCEPTION(eps <= 0.0, std::runtime_error, "eps must be positive");
  TEUCHOS_TEST_FOR_EXCEPTION(D   <= 0.0, std::runtime_error, "D must be positive");

  tt::solvers::SolverChoice which = tt::solvers::parseSolverChoice(solverName);

  const GO NglobNodes = GO(Nx)*GO(Ny);

  // ---- build owned/overlap maps and connectivity ----
  auto ownedNodeMap = Teuchos::rcp(new map_type(NglobNodes, 0, comm));
  ConnView ownedElemConn = tt::buildOwnedElementConnectivity(ownedNodeMap, Nx, Ny);
  auto overlapNodeMap = tt::buildOverlapNodeMap(ownedNodeMap, ownedElemConn);
  CoordView coordsOverlap = tt::buildCoordsStructured(overlapNodeMap, Nx, Ny, x0,x1,y0,y1);

  // ---- assemble overlap scalar blocks ----
  auto ov = assembleBlocksOverlap_Q1_MMS(overlapNodeMap, coordsOverlap, ownedElemConn,
                                         Nx, Ny, x0,x1,y0,y1,
                                         (SC)eps, (SC)D, (SC)alpha, (SC)zF);

  // ---- export overlap blocks to owned ----
  auto exportToOwned = [&](const Teuchos::RCP<const crs_type>& Aov) {
    return tt::exportToOwned(Aov, ownedNodeMap, overlapNodeMap);
  };

  auto A00 = exportToOwned(ov.A00_phi_phi);
  auto A01 = exportToOwned(ov.A01_phi_c);
  auto A10 = exportToOwned(ov.A10_c_phi);
  auto A11 = exportToOwned(ov.A11_c_c);

  // RHS and exact: export overlap->owned
  mv_type bphi(ownedNodeMap, 1), bc(ownedNodeMap, 1);
  mv_type phi_exact(ownedNodeMap, 1), c_exact(ownedNodeMap, 1);

  tt::exportMonolithicVector(*ov.bphi, bphi, *overlapNodeMap, *ownedNodeMap, Tpetra::ADD);
  tt::exportMonolithicVector(*ov.bc,   bc,   *overlapNodeMap, *ownedNodeMap, Tpetra::ADD);

  tt::exportMonolithicVector(*ov.phi_exact, phi_exact, *overlapNodeMap, *ownedNodeMap, Tpetra::INSERT);
  tt::exportMonolithicVector(*ov.c_exact,   c_exact,   *overlapNodeMap, *ownedNodeMap, Tpetra::INSERT);

  // ---- apply Dirichlet at subblock level (and set RHS boundary values) ----
  applyDirichletToScalarBlocksAndRHS(*A00,*A01,*A10,*A11, bphi, bc, Nx, Ny, x0,x1,y0,y1);

  // ---- build 2x2 blocked operator ----
  auto asThyra = [](const Teuchos::RCP<Tpetra::Operator<Teko::ST,Teko::LO,Teko::GO,Teko::NT>>& op){
    return Thyra::tpetraLinearOp<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(
      Thyra::tpetraVectorSpace<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(op->getRangeMap()),
      Thyra::tpetraVectorSpace<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(op->getDomainMap()),
      op);
  };

  auto bloOp = Teko::createBlockedOp();
  Teko::beginBlockFill(bloOp, 2, 2);

  Teko::setBlock(0,0,bloOp, asThyra(A00));
  Teko::setBlock(0,1,bloOp, asThyra(A01));
  Teko::setBlock(1,0,bloOp, asThyra(A10));
  Teko::setBlock(1,1,bloOp, asThyra(A11));

  Teko::endBlockFill(bloOp);

  // ---- merge to monolithic (for residual/L2 reporting and for Ifpack2 solve) ----
  using BlockedCrsMatrix = Xpetra::BlockedCrsMatrix<Teko::ST,Teko::LO,Teko::GO,Teko::NT>;
  auto blockedCrs = Teuchos::make_rcp<BlockedCrsMatrix>(bloOp, Teuchos::null);
  blockedCrs->fillComplete();

  auto monolithicX = blockedCrs->Merge();
  Teuchos::RCP<crs_type> Amono = Xpetra::toTpetra(monolithicX);
  auto Aop = Teuchos::rcp_dynamic_cast<Tpetra::Operator<SC,LO,GO,NO>>(Amono, true);

  // ---- build monolithic b and xexact ----
  const int nFields = 2;
  auto ownedMonoMap = Amono->getRowMap();

  mv_type bmono(ownedMonoMap, 1), xexact(ownedMonoMap, 1);
  {
    // pack bmono = [bphi; bc], xexact = [phi_exact; c_exact]
    auto bmh = bmono.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto xeh = xexact.getLocalViewHost(Tpetra::Access::OverwriteAll);

    auto bph = bphi.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto bch = bc.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto ph  = phi_exact.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto ch  = c_exact.getLocalViewHost(Tpetra::Access::ReadOnly);

    const LO nLocalNodes = (LO)ownedNodeMap->getLocalNumElements();
    for (LO l=0; l<nLocalNodes; ++l) {
      const GO nodeG = ownedNodeMap->getGlobalElement(l);
      const LO lphi = ownedMonoMap->getLocalElement(nodeG);
      const LO lc   = ownedMonoMap->getLocalElement(nodeG + NglobNodes);

      TEUCHOS_TEST_FOR_EXCEPTION(lphi==Teuchos::OrdinalTraits<LO>::invalid() ||
                                 lc==Teuchos::OrdinalTraits<LO>::invalid(),
                                 std::runtime_error, "monolithic map missing dof");

      bmh(lphi,0) = bph(l,0);
      bmh(lc,0)   = bch(l,0);

      xeh(lphi,0) = ph(l,0);
      xeh(lc,0)   = ch(l,0);
    }
  }

  // ---- build scalar mass matrix M for L2 errors ----
  auto km_for_mass = tt::buildKM_OverlapExported(
      Nx, Ny, x0,x1,y0,y1,
      [&](vec_type& v, const CoordView&){ v.putScalar(1.0); },
      [&](vec_type& v, const CoordView&){ v.putScalar(1.0); });
  auto M = km_for_mass.M;

  // ---- solve ----
  auto xmono = Teuchos::make_rcp<mv_type>(ownedMonoMap, 1);
  xmono->putScalar(0.0);

  auto report = [&](const std::string& label){
    auto res = explicitResidualNorm2(*Aop, *xmono, bmono);
    auto l2  = discreteL2ErrorMonolithic2Field(*M, *xmono, xexact);
    if (comm->getRank()==0) {
      std::cout << label
                << "  resid=" << res[0]
                << "  L2(phi)=" << l2.ephi
                << "  L2(c)="   << l2.ec
                << "  L2="      << l2.combined
                << "\n";
    }
  };

  auto b_rcp = Teuchos::rcpFromRef(bmono);

  switch (which) {
    case tt::solvers::SolverChoice::Ifpack2SchwarzRILUK: {
      auto Aop_mono = Teuchos::rcp_dynamic_cast<Tpetra::Operator<SC,LO,GO,NO>>(Amono, true);
      tt::solvers::solveWithIfpack2SchwarzRILUK_GMRES(Aop_mono, b_rcp, xmono, maxIters, tol, solverXml);
      report("Ifpack2 + GMRES:");
      break;
    }

    case tt::solvers::SolverChoice::TekoBGS: {
      tt::solvers::solveWithTekoBGS_GMRES(bloOp, b_rcp, xmono, maxIters, tol, solverXml);
      report("Teko BGS + GMRES:");
      break;
    }

    case tt::solvers::SolverChoice::TekoMonolithicAMG: {
      // For this PNP system, no special nullspace is required if Dirichlet is applied.
      // Still, BlockAMG can accept nullspace vectors; we omit them here.
      tt::solvers::solveWithTekoMonolithicAMG_GMRES(bloOp, b_rcp, xmono, maxIters, tol, solverXml);
      report("Teko Monolithic AMG + GMRES:");
      break;
    }
  }

  print_Teuchos_stacked_timer(Tpetra::getDefaultComm());

  return 0;
}