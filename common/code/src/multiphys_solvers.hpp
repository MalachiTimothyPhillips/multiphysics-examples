/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#pragma once

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Ifpack2_Factory.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Stratimikos_MueLuHelpers.hpp"

#include "Teko_TpetraOperatorWrapper.hpp"
#include "Teko_TpetraInverseFactoryOperator.hpp"
#include "Teko_InverseLibrary.hpp"
#include "Teko_PreconditionerFactory.hpp"
#include "Teko_CloneFactory.hpp"

#include "teko_amg.hpp" // multiphys::BlockAMGPreconditionerFactory

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <fstream>

namespace multiphys::solvers {

enum class SolverChoice {
  Ifpack2SchwarzRILUK,
  TekoBGS,
  TekoMonolithicAMG
};

inline SolverChoice parseSolverChoice(const std::string& s)
{
  if (s == "ifpack2" || s == "schwarz" || s == "dd-ilu") return SolverChoice::Ifpack2SchwarzRILUK;
  if (s == "teko-bgs" || s == "bgs") return SolverChoice::TekoBGS;
  if (s == "teko-amg" || s == "monolithic-amg" || s == "block-amg") return SolverChoice::TekoMonolithicAMG;

  throw std::runtime_error("Unknown --solver value '" + s +
                           "'. Use one of: ifpack2, teko-bgs, teko-amg");
}

// Load params from xml into an existing parameter list (overrides defaults).
inline void maybeOverrideParamsFromXml(Teuchos::ParameterList& pl,
                                      const std::string& xmlFile)
{
  if (xmlFile.empty()) return;

  std::ifstream in(xmlFile.c_str());
  if (!in.good())
    throw std::runtime_error("Could not open XML parameter file: " + xmlFile);

  Teuchos::updateParametersFromXmlFile(xmlFile, Teuchos::ptrFromRef(pl));
}


using ST = double;
using LO = int;
using GO = long long;
using NO = Tpetra::Map<>::node_type;

using MV = Tpetra::MultiVector<ST,LO,GO,NO>;
using OP = Tpetra::Operator<ST,LO,GO,NO>;

// ---------------- Belos GMRES helper ----------------
inline Teuchos::RCP<Belos::SolverManager<ST,MV,OP>>
makeGMRES(int maxIters, double tol, bool explicitResidualTest,
          int verbosity, int outputFreq, int numBlocks)
{
  Belos::SolverFactory<ST, MV, OP> factory;
  auto pl = Teuchos::rcp(new Teuchos::ParameterList);
  pl->set("Maximum Iterations", maxIters);
  pl->set("Convergence Tolerance", tol);
  pl->set("Verbosity", verbosity);
  pl->set("Output Frequency", outputFreq);
  pl->set("Output Style", 1);
  pl->set("Num Blocks", numBlocks);
  pl->set("Orthogonalization", "IMGS");
  if (explicitResidualTest) pl->set("Explicit Residual Test", true);
  return factory.create("GMRES", pl);
}

// ---------------- Ifpack2 Schwarz + RILUK ----------------
inline Teuchos::RCP<Ifpack2::Preconditioner<ST,LO,GO,NO>>
makeIfpack2SchwarzRILUK(const Teuchos::RCP<const OP>& A,
                        const std::string& xmlParams = "")
{
  auto Arow = Teuchos::rcp_dynamic_cast<const Tpetra::RowMatrix<ST,LO,GO,NO>>(A, true);

  Ifpack2::Factory factory;
  auto prec = factory.create<Tpetra::RowMatrix<ST,LO,GO,NO>>("SCHWARZ", Arow);

  Teuchos::ParameterList ifp;
  ifp.set("schwarz: combine mode", "Add");
  ifp.set("schwarz: overlap level", 1);
  ifp.set("schwarz: inner preconditioner name", "RILUK");
  ifp.set("schwarz: use reordering", true);

  auto& reorder_list = ifp.sublist("schwarz: reordering list", false);
  reorder_list.set("order_method", "rcm");

  auto& inner = ifp.sublist("schwarz: inner preconditioner parameters");
  inner.set("fact: iluk level-of-fill", 1);
  inner.set("fact: relax value", 0.0);
  inner.set("fact: absolute threshold", 0.0);
  inner.set("fact: relative threshold", 1.0);

  maybeOverrideParamsFromXml(ifp, xmlParams);

  prec->setParameters(ifp);
  prec->initialize();
  prec->compute();
  return prec;
}

inline void solveGMRES_RightPrec(const Teuchos::RCP<const OP>& A,
                                 const Teuchos::RCP<MV>& b,
                                 const Teuchos::RCP<MV>& x,
                                 const Teuchos::RCP<const OP>& rightPrec,
                                 int maxIters, double tol,
                                 bool explicitResidualTest = false)
{
  // Match your prior verbosity for the two “standard” runs
  const int verbosity = Belos::Errors + Belos::Warnings +
                        Belos::StatusTestDetails + Belos::IterationDetails;
  const int outputFreq = 1;
  const int numBlocks = 30;

  auto solver = makeGMRES(maxIters, tol, explicitResidualTest, verbosity, outputFreq, numBlocks);

  auto problem = Teuchos::rcp(new Belos::LinearProblem<ST,MV,OP>(A, x, b));
  if (rightPrec) problem->setRightPrec(rightPrec);
  const bool ok = problem->setProblem();
  TEUCHOS_TEST_FOR_EXCEPTION(!ok, std::runtime_error, "Belos::LinearProblem::setProblem() failed");
  solver->setProblem(problem);
  solver->solve();
}

// convenience: Ifpack2 solve
inline void solveWithIfpack2SchwarzRILUK_GMRES(const Teuchos::RCP<const OP>& A,
                                              const Teuchos::RCP<MV>& b,
                                              const Teuchos::RCP<MV>& x,
                                              int maxIters, double tol,
                                              const std::string& xmlParams = "")
{
  auto prec = makeIfpack2SchwarzRILUK(A, xmlParams);
  auto precOp = Teuchos::rcp_dynamic_cast<const OP>(prec, true);
  solveGMRES_RightPrec(A, b, x, precOp, maxIters, tol, /*explicitResidualTest=*/false);
}

// ---------------- Stratimikos/Teko builder ----------------
inline Teuchos::RCP<Stratimikos::DefaultLinearSolverBuilder> create_linear_solver_builder()
{
  auto strat = Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder);
  Stratimikos::enableMueLu<double, Teko::LO, Teko::GO, Teko::NT>(*strat);
  Teko::addToStratimikosBuilder(strat);
  return strat;
}

inline Teuchos::RCP<Teko::TpetraHelpers::InverseFactoryOperator>
construct_teko_preconditioner(const Teuchos::ParameterList& params,
                              const Teuchos::RCP<Teko::TpetraHelpers::TpetraOperatorWrapper>& Awrap)
{
  auto strat = create_linear_solver_builder();
  auto invLib = Teko::InverseLibrary::buildFromParameterList(params, strat);
  auto inverse = invLib->getInverseFactory("myInverse");

  auto prec = Teuchos::make_rcp<Teko::TpetraHelpers::InverseFactoryOperator>(inverse);
  prec->initInverse();

  auto Aop = Teuchos::rcp_static_cast<const OP>(Awrap);
  prec->rebuildInverseOperator(Aop);
  return prec;
}

// ---------------- Your original teko_params() ----------------
inline Teuchos::RCP<Teuchos::ParameterList> teko_params()
{
  auto params = Teuchos::make_rcp<Teuchos::ParameterList>();

  auto& bgs = params->sublist("myInverse");
  bgs.set("Type", "Block Gauss-Seidel");
  bgs.set("Inverse Type", "AMG");

  auto& amg = params->sublist("AMG");
  amg.set("Type", "MueLu");
  amg.set("verbosity", "none");
  amg.set("coarse: max size", 1000);
  amg.set("sa: use rowsumabs diagonal scaling", true);
  amg.set("smoother: type", "chebyshev");

  auto& smoother = amg.sublist("smoother: params");
  smoother.set("chebyshev: degree", 2);
  smoother.set("chebyshev: ratio eigenvalue", 7.0);
  smoother.set("chebyshev: min eigenvalue", 1.0);
  smoother.set("chebyshev: zero starting solution", true);
  smoother.set("chebyshev: use rowsumabs diagonal scaling", true);

  return params;
}

// ---------------- Your original teko_monolithic_amg_params() ----------------
inline Teuchos::RCP<Teuchos::ParameterList> teko_monolithic_amg_params()
{
  auto params = Teuchos::make_rcp<Teuchos::ParameterList>();

  auto& monolithic = params->sublist("myInverse");
  monolithic.set("Type", "Block AMG");

  auto& amg = monolithic.sublist("AMG Settings");
  amg.set("verbosity", "none");
  amg.set("coarse: max size", 1000);
  amg.set("smoother: type", "teko");

  amg.sublist("subblockList0").set("coarse: max size", 1000);
  amg.sublist("subblockList0").set("sa: use rowsumabs diagonal scaling", true);

  amg.sublist("subblockList1").set("coarse: max size", 1000);
  amg.sublist("subblockList1").set("sa: use rowsumabs diagonal scaling", true);

  auto& smoother = amg.sublist("smoother: params");
  smoother.set("Inverse Type", "bgs");

  auto& tekoSettings = smoother.sublist("Inverse Factory Library");
  tekoSettings.sublist("bgs").set("Type", "Block Gauss-Seidel");
  tekoSettings.sublist("bgs").set("Inverse Type", "Chebyshev");

  tekoSettings.sublist("Chebyshev").set("Type", "Ifpack2");
  tekoSettings.sublist("Chebyshev").set("Prec Type", "CHEBYSHEV");
  auto& ifp = tekoSettings.sublist("Chebyshev").sublist("Ifpack2 Settings");
  ifp.set("chebyshev: degree", 2);
  ifp.set("chebyshev: ratio eigenvalue", 7.0);
  ifp.set("chebyshev: min eigenvalue", 1.0);
  ifp.set("chebyshev: zero starting solution", true);

  return params;
}

// ---------------- Solve with Teko BGS ----------------
inline void solveWithTekoBGS_GMRES(const Teko::BlockedLinearOp& blo,
                                  const Teuchos::RCP<MV>& b,
                                  const Teuchos::RCP<MV>& x,
                                  int maxIters, double tol,
                                  const std::string& xmlParams = "",
                                  std::function<void(Teuchos::ParameterList&)> set_user_data = [](Teuchos::ParameterList&){})
{
  Teko::LinearOp A_lo = blo;
  auto Awrap = Teuchos::make_rcp<Teko::TpetraHelpers::TpetraOperatorWrapper>(A_lo);

  Teuchos::ParameterList params = *teko_params(); // defaults
  maybeOverrideParamsFromXml(params, xmlParams);

  set_user_data(params);

  auto prec = construct_teko_preconditioner(params, Awrap);

  auto precOp = Teuchos::rcp_static_cast<const OP>(prec);
  auto Aop    = Teuchos::rcp_static_cast<const OP>(Awrap);

  solveGMRES_RightPrec(Aop, b, x, precOp, maxIters, tol, /*explicitResidualTest=*/false);
}

// ---------------- Solve with custom Block AMG (“Monolithic AMG”) ----------------
inline void ensureBlockAMGFactoryRegistered()
{
  static bool registered = false;
  if (registered) return;

  Teuchos::RCP<Teko::Cloneable> clone =
      Teuchos::rcp(new Teko::AutoClone<multiphys::BlockAMGPreconditionerFactory>());
  Teko::PreconditionerFactory::addPreconditionerFactory("Block AMG", clone);

  registered = true;
}

inline void solveWithTekoMonolithicAMG_GMRES(const Teko::BlockedLinearOp& blo,
                                            const Teuchos::RCP<MV>& b,
                                            const Teuchos::RCP<MV>& x,
                                            int maxIters, double tol,
                                            const std::string& xmlParams = "",
                                            std::function<void(Teuchos::ParameterList&)> set_user_data = [](Teuchos::ParameterList&){})
{
  ensureBlockAMGFactoryRegistered();

  Teko::LinearOp A_lo = blo;
  auto Awrap = Teuchos::make_rcp<Teko::TpetraHelpers::TpetraOperatorWrapper>(A_lo);

  Teuchos::ParameterList params = *teko_monolithic_amg_params(); // defaults
  maybeOverrideParamsFromXml(params, xmlParams);

  set_user_data(params);

  auto prec = construct_teko_preconditioner(params, Awrap);

  auto precOp = Teuchos::rcp_static_cast<const OP>(prec);
  auto Aop    = Teuchos::rcp_static_cast<const OP>(Awrap);

  solveGMRES_RightPrec(Aop, b, x, precOp, maxIters, tol, /*explicitResidualTest=*/false);
}

} // namespace multiphys::solvers