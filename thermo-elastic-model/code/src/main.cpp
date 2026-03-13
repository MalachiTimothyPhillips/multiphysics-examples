/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

// main.cpp (thermoelastic benchmark, 3-field monolithic: ux, uy, T)

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>

#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>

#include "Teko_Utilities.hpp"

#include "multiphys_mesh.hpp"
#include "multiphys_q1_assembly.hpp"
#include "multiphys_transfer.hpp"
#include "multiphys_solvers.hpp"

#include "multiphys_dof.hpp"
#include "multiphys_dirichlet.hpp"
#include "multiphys_thermoelastic_q1.hpp"
#include "multiphys_mms_thermoelastic.hpp"
#include "multiphys_nullspace.hpp"

#include <Teuchos_CommandLineProcessor.hpp>

namespace {

// ---------- residual norm helper ----------
template<class SC, class LO, class GO, class NO>
Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType>
explicitResidualNorm2(const Tpetra::Operator<SC,LO,GO,NO>& A,
                      const Tpetra::MultiVector<SC,LO,GO,NO>& x,
                      const Tpetra::MultiVector<SC,LO,GO,NO>& b)
{
  using MV = Tpetra::MultiVector<SC,LO,GO,NO>;
  MV r(b.getMap(), b.getNumVectors());
  A.apply(x, r);
  r.update(1.0, b, -1.0);
  Teuchos::Array<typename Teuchos::ScalarTraits<SC>::magnitudeType> n(b.getNumVectors());
  r.norm2(n());
  return n;
}

// ---------- L2 error (3-field) using scalar mass matrix M for each field ----------
template<class SC, class LO, class GO, class NO>
struct L2Error3Field {
  typename Teuchos::ScalarTraits<SC>::magnitudeType ex;
  typename Teuchos::ScalarTraits<SC>::magnitudeType ey;
  typename Teuchos::ScalarTraits<SC>::magnitudeType eT;
  typename Teuchos::ScalarTraits<SC>::magnitudeType combined;
};

template<class SC, class LO, class GO, class NO>
L2Error3Field<SC,LO,GO,NO>
discreteL2ErrorMonolithic3Field(const Tpetra::CrsMatrix<SC,LO,GO,NO>& M,
                                const Tpetra::MultiVector<SC,LO,GO,NO>& x,
                                const Tpetra::MultiVector<SC,LO,GO,NO>& xe)
{
  using MV = Tpetra::MultiVector<SC,LO,GO,NO>;
  using mag_type = typename Teuchos::ScalarTraits<SC>::magnitudeType;

  const LO nLocalMono = (LO)x.getLocalLength();
  TEUCHOS_TEST_FOR_EXCEPTION(nLocalMono % 3 != 0, std::runtime_error,
                             "discreteL2ErrorMonolithic3Field: local length not divisible by 3");
  const LO nLocalNode = nLocalMono / 3;

  MV ex(M.getRowMap(), 1), ey(M.getRowMap(), 1), eT(M.getRowMap(), 1);
  {
    auto xv  = x.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto xev = xe.getLocalViewHost(Tpetra::Access::ReadOnly);

    auto exv = ex.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto eyv = ey.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto eTv = eT.getLocalViewHost(Tpetra::Access::OverwriteAll);

    for (LO i=0;i<nLocalNode;++i) {
      exv(i,0) = xv(i,0)                - xev(i,0);                 // ux
      eyv(i,0) = xv(i+nLocalNode,0)     - xev(i+nLocalNode,0);      // uy
      eTv(i,0) = xv(i+2*nLocalNode,0)   - xev(i+2*nLocalNode,0);    // T
    }
  }

  MV Mex(M.getRowMap(), 1), Mey(M.getRowMap(), 1), MeT(M.getRowMap(), 1);
  M.apply(ex, Mex);
  M.apply(ey, Mey);
  M.apply(eT, MeT);

  Teuchos::Array<mag_type> tmp(1);

  ex.dot(Mex, tmp()); mag_type ex2 = tmp[0];
  ey.dot(Mey, tmp()); mag_type ey2 = tmp[0];
  eT.dot(MeT, tmp()); mag_type eT2 = tmp[0];

  // guard against tiny negative due to roundoff
  ex2 = (ex2 < mag_type(0) ? mag_type(0) : ex2);
  ey2 = (ey2 < mag_type(0) ? mag_type(0) : ey2);
  eT2 = (eT2 < mag_type(0) ? mag_type(0) : eT2);

  return {std::sqrt(ex2), std::sqrt(ey2), std::sqrt(eT2), std::sqrt(ex2+ey2+eT2)};
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

  // ---- PDE/physics defaults ----
  double lambda = 1.0;
  double mu     = 1.0;

  double kappa  = 1.0;
  double rhoCp  = 1.0;
  double wx     = 1.0;
  double wy     = 0.0;

  double beta   = 1e-2;
  double eta    = 1e-2;

  // ---- solver defaults ----
  std::string solverName = "ifpack2";
  std::string solverXml  = "";
  int maxIters = 200;
  double tol = 1e-10;

  // ---- parse cmdline ----
  Teuchos::CommandLineProcessor clp(false, true);

  clp.setOption("Nx", &Nx, "Number of nodes in x direction (structured grid)");
  clp.setOption("Ny", &Ny, "Number of nodes in y direction (structured grid)");

  clp.setOption("lambda", &lambda, "Elasticity Lame parameter lambda");
  clp.setOption("mu",     &mu,     "Elasticity Lame parameter mu");

  clp.setOption("kappa",  &kappa,  "Thermal conductivity");
  clp.setOption("rhoCp",  &rhoCp,  "rho*c_p scaling for heat advection");
  clp.setOption("wx",     &wx,     "Heat advection velocity x-component");
  clp.setOption("wy",     &wy,     "Heat advection velocity y-component");

  clp.setOption("beta",   &beta,   "Coupling: mechanics <- T");
  clp.setOption("eta",    &eta,    "Coupling: heat <- div(u)");

  clp.setOption("solver", &solverName, "Solver to run: ifpack2 | teko-bgs | teko-amg");
  clp.setOption("solver-xml", &solverXml,
                "Optional XML file to override the solver parameter list for the chosen solver");
  clp.setOption("max-iters", &maxIters, "GMRES maximum iterations");
  clp.setOption("tol", &tol, "GMRES convergence tolerance");

  auto rc = clp.parse(argc, argv);
  if (rc != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    return (rc == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED ? 0 : 1);

  // ---- sanity checks ----
  TEUCHOS_TEST_FOR_EXCEPTION(Nx < 2 || Ny < 2, std::runtime_error, "Nx and Ny must be >= 2");
  TEUCHOS_TEST_FOR_EXCEPTION(mu <= 0.0, std::runtime_error, "mu must be positive");
  TEUCHOS_TEST_FOR_EXCEPTION(kappa <= 0.0, std::runtime_error, "kappa must be positive");
  TEUCHOS_TEST_FOR_EXCEPTION(rhoCp < 0.0, std::runtime_error, "rhoCp must be nonnegative");

  tt::solvers::SolverChoice which = tt::solvers::parseSolverChoice(solverName);

  // ---- assemble block operators on owned map ----
  tt::thermoelastic::Params p;
  p.lambda = (SC)lambda;
  p.mu     = (SC)mu;
  p.kappa  = (SC)kappa;
  p.rhoCp  = (SC)rhoCp;
  p.wx     = (SC)wx;
  p.wy     = (SC)wy;
  p.beta   = (SC)beta;
  p.eta    = (SC)eta;

  auto blks = tt::thermoelastic::buildThermoelasticBlocks_OverlapExported(Nx, Ny, x0,x1,y0,y1, p);

  // Also build a scalar mass matrix M for L2 errors (reuse existing Q1 code):
  // We can use buildKM_OverlapExported and just grab M; k values don't matter for M.
  auto km_for_mass = tt::buildKM_OverlapExported(
      Nx, Ny, x0,x1,y0,y1,
      [&](vec_type& v, const CoordView&){ v.putScalar(1.0); },
      [&](vec_type& v, const CoordView&){ v.putScalar(1.0); });
  auto M = km_for_mass.M;

  if (comm->getRank()==0) {
    std::cout << "Thermoelastic blocks built:\n"
              << "  Kxx nnz=" << blks.Kxx->getGlobalNumEntries() << "\n"
              << "  AT  nnz=" << blks.AT ->getGlobalNumEntries() << "\n"
              << "  Mass M nnz=" << M->getGlobalNumEntries() << "\n";
  }

  // ---- build 2x2 blocked operator ----
  auto asThyra = [](const Teuchos::RCP<Tpetra::Operator<Teko::ST,Teko::LO,Teko::GO,Teko::NT>>& op){
    return Thyra::tpetraLinearOp<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(
      Thyra::tpetraVectorSpace<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(op->getRangeMap()),
      Thyra::tpetraVectorSpace<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(op->getDomainMap()),
      op);
  };

  auto bloOp = Teko::createBlockedOp();
  Teko::beginBlockFill(bloOp, 2, 2);

  Teko::setBlock(0,0,bloOp, asThyra(blks.Auu));
  Teko::setBlock(0,1,bloOp, asThyra(blks.AuT));
  Teko::setBlock(1,0,bloOp, asThyra(blks.ATu));
  Teko::setBlock(1,1,bloOp, asThyra(blks.AT));

  Teko::endBlockFill(bloOp);

  // ---- merge to monolithic ----
  using BlockedCrsMatrix = Xpetra::BlockedCrsMatrix<Teko::ST,Teko::LO,Teko::GO,Teko::NT>;
  auto blockedCrs = Teuchos::make_rcp<BlockedCrsMatrix>(bloOp, Teuchos::null);
  blockedCrs->fillComplete();
  auto monolithicX = blockedCrs->Merge();
  Teuchos::RCP<tt::crs_type> Amono = Xpetra::toTpetra(monolithicX);

  if (comm->getRank()==0)
    std::cout << "Monolithic A nnz=" << Amono->getGlobalNumEntries() << "\n";

  // ---- Build overlap maps for MMS ----
  const GO NglobNodes = GO(Nx) * GO(Ny);

  auto ownedNodeMap = blks.Kxx->getRowMap();
  auto ownedElemConn = tt::buildOwnedElementConnectivity(ownedNodeMap, Nx, Ny);
  auto ownedCoords  = tt::buildCoordsStructured(ownedNodeMap, Nx, Ny, x0,x1,y0,y1);

  auto overlapNodeMap = tt::buildOverlapNodeMap(ownedNodeMap, ownedElemConn);
  auto coordsOverlap  = tt::buildCoordsStructured(overlapNodeMap, Nx, Ny, x0,x1,y0,y1);

  const int nFields = 3;
  auto overlapMonoMap = tt::buildMonolithicMapNFieldFromNodeMap(overlapNodeMap, NglobNodes, nFields);
  auto ownedMonoMap   = Amono->getRowMap();

  // ---- MMS assembly on overlap monolithic map ----
  Tpetra::MultiVector<SC,LO,GO,NO> b_ov(overlapMonoMap, 1), xexact_ov(overlapMonoMap, 1);

  tt::mms_te::assembleMonolithicRHS_andExact_onOverlap(
      overlapNodeMap, overlapMonoMap, coordsOverlap, ownedElemConn,
      b_ov, xexact_ov, Nx, Ny,
      p.lambda, p.mu,
      p.kappa, p.rhoCp, p.wx, p.wy,
      p.beta, p.eta);

  // ---- Export overlap -> owned ----
  Tpetra::MultiVector<SC,LO,GO,NO> b(ownedMonoMap, 1), xexact(ownedMonoMap, 1);
  tt::exportMonolithicVector(b_ov, b, *overlapMonoMap, *ownedMonoMap, Tpetra::ADD);
  tt::exportMonolithicVector(xexact_ov, xexact, *overlapMonoMap, *ownedMonoMap, Tpetra::INSERT);

  // ---- Apply homogeneous Dirichlet (N-field generalized) ----
  tt::applyHomogeneousDirichlet_MonolithicNField(*Amono, &b, Nx, Ny, nFields);

  auto Aop = Teuchos::rcp_dynamic_cast<Tpetra::Operator<SC,LO,GO,NO>>(Amono, true);

  // ---- Build rigid body modes for AMG solver ----
  SC xcen = 0.5*(x0+x1);
  SC ycen = 0.5*(y0+y1);
  auto uMap = blks.Auu->getRowMap();
  auto rbm = tt::buildRigidBodyModes2D(ownedNodeMap, uMap, ownedCoords, NglobNodes, xcen, ycen);

  // ---- Solve ----
  auto b_rcp      = Teuchos::rcpFromRef(b);
  auto x_rcp      = Teuchos::make_rcp<Tpetra::MultiVector<SC,LO,GO,NO>>(ownedMonoMap, 1);
  auto xexact_rcp = Teuchos::rcpFromRef(xexact);
  x_rcp->putScalar(0.0);

  auto report = [&](const std::string& label){
    auto res = explicitResidualNorm2(*Aop, *x_rcp, *b_rcp);
    auto l2  = discreteL2ErrorMonolithic3Field(*M, *x_rcp, *xexact_rcp);

    if (comm->getRank()==0) {
      std::cout << label
                << "  resid=" << res[0]
                << "  L2(ux)=" << l2.ex
                << "  L2(uy)=" << l2.ey
                << "  L2(T)="  << l2.eT
                << "  L2="     << l2.combined
                << "\n";
    }
  };

  // TODO: this is a bit of a kludge that relies on the exact names in the MueLu params
  auto set_user_data = [&](Teuchos::ParameterList& params){
    std::cout << params << "\n";
    if(!params.isSublist("Elasticity")) return;
    auto & elasticity = params.sublist("Elasticity");
    elasticity.sublist("user data").set("Nullspace", rbm);
  };

  auto set_user_data_monolithic = [&](Teuchos::ParameterList& params){
    if(!params.isSublist("myInverse")) return;
    auto & blockAMG = params.sublist("myInverse");
    blockAMG.sublist("user data").set("Nullspace0", Xpetra::toXpetra(rbm)); // set up nullspace for elasticity
  };

  switch (which) {
    case tt::solvers::SolverChoice::Ifpack2SchwarzRILUK:
      tt::solvers::solveWithIfpack2SchwarzRILUK_GMRES(Aop, b_rcp, x_rcp, maxIters, tol, solverXml);
      report("Ifpack2 + GMRES:");
      break;

    case tt::solvers::SolverChoice::TekoBGS:
      tt::solvers::solveWithTekoBGS_GMRES(bloOp, b_rcp, x_rcp, maxIters, tol, solverXml, set_user_data);
      report("Block Gauss-Seidel + GMRES:");
      break;

    // TODO: we should be able to consolidate this code path with the one above, provided the user gives params
    case tt::solvers::SolverChoice::TekoMonolithicAMG:
      tt::solvers::solveWithTekoMonolithicAMG_GMRES(bloOp, b_rcp, x_rcp, maxIters, tol, solverXml, set_user_data_monolithic);
      report("Monolithic AMG + GMRES:");
      break;
  }

  return 0;
}