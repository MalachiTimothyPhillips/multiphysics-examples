#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>

#include <TpetraExt_MatrixMatrix.hpp>

#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Ifpack2_Factory.hpp>

#include <Amesos2.hpp>

#include "Teko_Utilities.hpp"
#include "Teko_TpetraOperatorWrapper.hpp"
#include "Teko_TpetraInverseFactoryOperator.hpp"
#include "Teko_InverseLibrary.hpp"
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Stratimikos_MueLuHelpers.hpp"

#include "Solv_TekoAMG.hpp"

#include "tt_mesh.hpp"
#include "tt_q1_assembly.hpp"
#include "tt_mms.hpp"
#include "tt_dirichlet.hpp"
#include "tt_dof.hpp"
#include "tt_transfer.hpp"
#include "tt_solvers.hpp"
#include "tt_dirichlet_blocks.hpp"

#include <Teuchos_CommandLineProcessor.hpp>

namespace {

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

// same discreteL2ErrorMonolithic2Field you had, kept as-is (it’s useful)
template<class SC, class LO, class GO, class NO>
struct L2Error2Field {
  typename Teuchos::ScalarTraits<SC>::magnitudeType ee;
  typename Teuchos::ScalarTraits<SC>::magnitudeType el;
  typename Teuchos::ScalarTraits<SC>::magnitudeType combined;
};

template<class SC, class LO, class GO, class NO>
L2Error2Field<SC,LO,GO,NO>
discreteL2ErrorMonolithic2Field(const Tpetra::CrsMatrix<SC,LO,GO,NO>& M,
                                const Tpetra::MultiVector<SC,LO,GO,NO>& x,
                                const Tpetra::MultiVector<SC,LO,GO,NO>& xe)
{
  using MV = Tpetra::MultiVector<SC,LO,GO,NO>;
  using mag_type = typename Teuchos::ScalarTraits<SC>::magnitudeType;

  const LO nLocalMono = (LO)x.getLocalLength();
  const LO nLocalNode = nLocalMono/2;

  MV ee(M.getRowMap(), 1), el(M.getRowMap(), 1);
  {
    auto xv  = x.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto xev = xe.getLocalViewHost(Tpetra::Access::ReadOnly);
    auto eev = ee.getLocalViewHost(Tpetra::Access::OverwriteAll);
    auto elv = el.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (LO i=0;i<nLocalNode;++i) {
      eev(i,0) = xv(i,0)            - xev(i,0);
      elv(i,0) = xv(i+nLocalNode,0) - xev(i+nLocalNode,0);
    }
  }

  MV Mee(M.getRowMap(), 1), Mel(M.getRowMap(), 1);
  M.apply(ee, Mee);
  M.apply(el, Mel);

  Teuchos::Array<mag_type> tmp(1);
  ee.dot(Mee, tmp()); mag_type ee2 = tmp[0];
  el.dot(Mel, tmp()); mag_type el2 = tmp[0];
  ee2 = (ee2 < mag_type(0) ? mag_type(0) : ee2);
  el2 = (el2 < mag_type(0) ? mag_type(0) : el2);

  return {std::sqrt(ee2), std::sqrt(el2), std::sqrt(ee2+el2)};
}

template<class ST, class LO, class GO, class NO>
void solveGMRESRightPrec(const Teuchos::RCP<Tpetra::Operator<ST,LO,GO,NO>>& A,
                         const Teuchos::RCP<Tpetra::MultiVector<ST,LO,GO,NO>>& b,
                         const Teuchos::RCP<Tpetra::MultiVector<ST,LO,GO,NO>>& x,
                         const Teuchos::RCP<Tpetra::Operator<ST,LO,GO,NO>>& rightPrec,
                         int maxIters, double tol)
{
  using MV = Tpetra::MultiVector<ST,LO,GO,NO>;
  using OP = Tpetra::Operator<ST,LO,GO,NO>;

  Belos::SolverFactory<ST, MV, OP> fac;
  auto pl = Teuchos::rcp(new Teuchos::ParameterList);
  pl->set("Maximum Iterations", maxIters);
  pl->set("Convergence Tolerance", tol);
  pl->set("Num Blocks", 30);
  pl->set("Orthogonalization", "IMGS");

  pl->set("Verbosity", Belos::Errors + Belos::Warnings + Belos::StatusTestDetails + Belos::IterationDetails);
  pl->set("Output Frequency", 1);
  pl->set("Output Style", 1);

  auto solver = fac.create("GMRES", pl);
  auto problem = Teuchos::rcp(new Belos::LinearProblem<ST,MV,OP>(A, x, b));
  if (rightPrec) problem->setRightPrec(rightPrec);
  if (!problem->setProblem()) throw std::runtime_error("Belos problem setup failed");
  solver->setProblem(problem);
  solver->solve();
}

} // namespace

int main(int argc, char** argv)
{
  Tpetra::ScopeGuard scope(&argc, &argv);
  using namespace tt;

  int Nx = 81*8;
  int Ny = 81*8;
  double g = 1e-3;

  const SC x0=0.0, x1=1.0, y0=0.0, y1=1.0;

  auto comm = Tpetra::getDefaultComm();

  std::string solverName = "ifpack2";
  std::string solverXml  = "";
  int maxIters = 200;
  double tol = 1e-10;

  Teuchos::CommandLineProcessor clp(false, true);
  clp.setOption("Nx", &Nx, "Number of nodes in x direction (structured grid)");
  clp.setOption("Ny", &Ny, "Number of nodes in y direction (structured grid)");
  clp.setOption("g",  &g,  "Coupling coefficient");
  
  clp.setOption("solver", &solverName,
                "Solver to run: ifpack2 | teko-bgs | teko-amg");
  clp.setOption("solver-xml", &solverXml,
                "Optional XML file to override the solver parameter list for the chosen solver");
  clp.setOption("max-iters", &maxIters, "GMRES maximum iterations");
  clp.setOption("tol", &tol, "GMRES convergence tolerance");
  
  auto rc = clp.parse(argc, argv);
  if (rc != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    return (rc == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED ? 0 : 1);
  
  // sanity checks
  TEUCHOS_TEST_FOR_EXCEPTION(Nx < 2 || Ny < 2, std::runtime_error, "Nx and Ny must be >= 2");

  tt::solvers::SolverChoice which = tt::solvers::parseSolverChoice(solverName);

  const SC ke = 1.0;
  const SC kl = 0.5;

  // 1) Build Ke, Kl, M (owned)
  auto km = tt::buildKM_OverlapExported(
      Nx, Ny, x0,x1,y0,y1,
      [&](vec_type& v, const CoordView&){ v.putScalar(ke); },
      [&](vec_type& v, const CoordView&){ v.putScalar(kl); });

  if (comm->getRank()==0) {
    std::cout << "KM built: Ke nnz=" << km.Ke->getGlobalNumEntries()
              << " M nnz=" << km.M->getGlobalNumEntries() << "\n";
  }

  // 2) Form block matrices (owned)
  using crs_t = tt::crs_type;
  Teuchos::RCP<crs_t> A11, A22;
  Tpetra::MatrixMatrix::Add(*km.Ke, false, 1.0, *km.M, false, g, A11); A11->fillComplete();
  Tpetra::MatrixMatrix::Add(*km.Kl, false, 1.0, *km.M, false, g, A22); A22->fillComplete();

  auto A12 = Teuchos::rcp(new crs_t(*km.M, Teuchos::DataAccess::Copy));
  auto A21 = Teuchos::rcp(new crs_t(*km.M, Teuchos::DataAccess::Copy));
  A12->scale(-g); A21->scale(-g);

  // Apply boundary conditions 
  auto boundary = tt::boundaryNodeGIDs(Nx, Ny);

  tt::applyDirichletRows_DiagBlock(*A11, boundary());
  tt::applyDirichletRows_DiagBlock(*A22, boundary());
  tt::applyDirichletRows_OffDiagBlock(*A12, boundary());
  tt::applyDirichletRows_OffDiagBlock(*A21, boundary());

  // 3) Build blocked operator and merged monolithic matrix
  auto bloOp = Teko::createBlockedOp();
  Teko::beginBlockFill(bloOp, 2, 2);

  auto asThyra = [](const Teuchos::RCP<Tpetra::Operator<Teko::ST,Teko::LO,Teko::GO,Teko::NT>>& op){
    return Thyra::tpetraLinearOp<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(
      Thyra::tpetraVectorSpace<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(op->getRangeMap()),
      Thyra::tpetraVectorSpace<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(op->getDomainMap()),
      op);
  };

  Teko::setBlock(0,0,bloOp, asThyra(A11));
  Teko::setBlock(0,1,bloOp, asThyra(A12));
  Teko::setBlock(1,0,bloOp, asThyra(A21));
  Teko::setBlock(1,1,bloOp, asThyra(A22));
  Teko::endBlockFill(bloOp);

  using BlockedCrsMatrix = Xpetra::BlockedCrsMatrix<Teko::ST,Teko::LO,Teko::GO,Teko::NT>;
  auto blockedCrs = Teuchos::make_rcp<BlockedCrsMatrix>(bloOp, Teuchos::null);
  blockedCrs->fillComplete();
  auto monolithicX = blockedCrs->Merge();
  Teuchos::RCP<crs_t> Amono = Xpetra::toTpetra(monolithicX);

  if (comm->getRank()==0)
    std::cout << "Monolithic A nnz=" << Amono->getGlobalNumEntries() << "\n";

  const GO NglobNodes = GO(Nx) * GO(Ny);

  // 4) Build overlap maps for MMS (true ghosting)

  auto ownedNodeMap = km.M->getRowMap();
  auto ownedElemConn = tt::buildOwnedElementConnectivity(ownedNodeMap, Nx, Ny);

  auto overlapNodeMap = tt::buildOverlapNodeMap(ownedNodeMap, ownedElemConn);
  auto coordsOverlap  = tt::buildCoordsStructured(overlapNodeMap, Nx, Ny, x0,x1,y0,y1);

  auto overlapMonoMap = tt::buildMonolithicMap2FieldFromNodeMap(overlapNodeMap, NglobNodes);

  // Owned monolithic map comes from the merged matrix
  auto ownedMonoMap = Amono->getRowMap();

  // 5) Assemble MMS on overlap monolithic map
  Tpetra::MultiVector<SC,LO,GO,NO> b_ov(overlapMonoMap, 1), xexact_ov(overlapMonoMap, 1);
  tt::mms::assembleMonolithicRHS_andExact_onOverlap(
      overlapNodeMap, overlapMonoMap, coordsOverlap, ownedElemConn,
      b_ov, xexact_ov, Nx, Ny, ke, kl, g);

  // 6) Export overlap -> owned
  Tpetra::MultiVector<SC,LO,GO,NO> b(ownedMonoMap, 1), xexact(ownedMonoMap, 1), x(ownedMonoMap, 1);

  // RHS assembled on overlap should be ADD-exported
  tt::exportMonolithicVector(b_ov, b, *overlapMonoMap, *ownedMonoMap, Tpetra::ADD);

  // Exact nodal samples are duplicates-consistent -> INSERT is fine (ADD also works but noisier)
  tt::exportMonolithicVector(xexact_ov, xexact, *overlapMonoMap, *ownedMonoMap, Tpetra::INSERT);

  // Apply Dirichlet to monolithic system + owned RHS
  tt::applyHomogeneousDirichlet_Monolithic2Field(*Amono, &b, Nx, Ny);

  // 7) Solve (example: Ifpack2 Schwarz+RILUK)
  x.putScalar(0.0);

  auto Aop = Teuchos::rcp_dynamic_cast<Tpetra::Operator<SC,LO,GO,NO>>(Amono, true);

  auto b_rcp = Teuchos::rcpFromRef(b);
  auto x_rcp = Teuchos::make_rcp<Tpetra::MultiVector<SC,LO,GO,NO>>(ownedMonoMap, 1);
  auto xexact_rcp = Teuchos::rcpFromRef(xexact);
  
  // reporting helper (reuse your discrete L2)
  auto report = [&](const std::string& label){
    auto res = explicitResidualNorm2(*Aop, *x_rcp, *b_rcp);
    auto l2  = discreteL2ErrorMonolithic2Field(*km.M, *x_rcp, *xexact_rcp);
  
    if (comm->getRank()==0) {
      std::cout << label
                << "  resid=" << res[0]
                << "  L2(Te)=" << l2.ee
                << "  L2(Tl)=" << l2.el
                << "  L2=" << l2.combined << "\n";
    }
  };
  
  x_rcp->putScalar(0.0);

  switch (which) {
    case tt::solvers::SolverChoice::Ifpack2SchwarzRILUK:
      tt::solvers::solveWithIfpack2SchwarzRILUK_GMRES(Aop, b_rcp, x_rcp, maxIters, tol, solverXml);
      report("Ifpack2 + GMRES:");
      break;
  
    case tt::solvers::SolverChoice::TekoBGS:
      tt::solvers::solveWithTekoBGS_GMRES(bloOp, b_rcp, x_rcp, maxIters, tol, solverXml);
      report("Block Gauss-Seidel + GMRES:");
      break;
  
    case tt::solvers::SolverChoice::TekoMonolithicAMG:
      tt::solvers::solveWithTekoMonolithicAMG_GMRES(bloOp, b_rcp, x_rcp, maxIters, tol, solverXml);
      report("Monolithic AMG + GMRES:");
      break;
  }

  return 0;
}