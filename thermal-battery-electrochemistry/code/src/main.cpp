/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>

#include <Xpetra_BlockedCrsMatrix.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_TpetraCrsMatrix.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Amesos2.hpp>

#include "Teko_Utilities.hpp"
#include "Teko_BlockedTpetraOperator.hpp"

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <cmath>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <string>

#include "multiphys_mesh.hpp"
#include "multiphys_transfer.hpp"
#include "multiphys_solvers.hpp"
#include "multiphys_dirichlet_blocks.hpp"

namespace {

using SC = multiphys::SC;
using LO = multiphys::LO;
using GO = multiphys::GO;
using NO = multiphys::NO;
using map_type = multiphys::map_type;
using crs_type = Tpetra::CrsMatrix<SC,LO,GO,NO>;
using mv_type  = Tpetra::MultiVector<SC,LO,GO,NO>;
using vec_type = Tpetra::Vector<SC,LO,GO,NO>;
using comm_type = Teuchos::Comm<int>;

struct BatteryMesh
{
  int numCells{10};
  int nxPerCell{20};
  int ny{16};

  SC lx{1.0};
  SC ly{1.0};

  int nx() const { return numCells * nxPerCell; }
  int NxNodes() const { return nx() + 1; }
  int NyNodes() const { return ny + 1; }

  GO num_scalar_dofs() const { return static_cast<GO>(NxNodes()) * static_cast<GO>(NyNodes()); }
  GO num_total_dofs() const { return 2 * num_scalar_dofs(); }

  SC hx() const { return lx / static_cast<SC>(nx()); }
  SC hy() const { return ly / static_cast<SC>(ny); }
  SC cell_width() const { return lx / static_cast<SC>(numCells); }

  GO node_gid(const int i, const int j) const { return static_cast<GO>(j * NxNodes() + i); }

  std::pair<int, int> ij_from_scalar_gid(const GO scalarGid) const
  {
    const int i = static_cast<int>(scalarGid % NxNodes());
    const int j = static_cast<int>(scalarGid / NxNodes());
    return {i, j};
  }

  std::pair<SC, SC> xy_from_scalar_gid(const GO scalarGid) const
  {
    const auto [i, j] = ij_from_scalar_gid(scalarGid);
    return {static_cast<SC>(i) * hx(), static_cast<SC>(j) * hy()};
  }

  int cell_index_from_x(const SC x) const
  {
    const int cell = static_cast<int>(std::floor(x / cell_width()));
    return std::min(std::max(cell, 0), numCells - 1);
  }

  SC local_xi_from_x(const SC x) const
  {
    const int cell = cell_index_from_x(x);
    const SC xLeft = static_cast<SC>(cell) * cell_width();
    return (x - xLeft) / cell_width();
  }

  GO liquid_pin_scalar_gid() const { return node_gid(nx() / 2, ny / 2); }
};

enum class BatteryRegion : int
{
  AnodeCollector = 0,
  Anode = 1,
  Separator = 2,
  Cathode = 3,
  CathodeCollector = 4,
  Inclusion = 5
};

struct MaterialCoeffs
{
  BatteryRegion region;
  SC sigma_s;
  SC kappa_l;
  SC g;
};

bool in_local_ellipse(const SC xi, const SC y, const SC cxi, const SC cy, const SC rxi, const SC ry)
{
  const SC dxi = (xi - cxi) / rxi;
  const SC dy = (y - cy) / ry;
  return dxi * dxi + dy * dy <= 1.0;
}

SC shifted_y(const SC y0, const SC shift)
{
  const SC y = std::fmod(y0 + shift, 1.0);
  return y < 0.0 ? y + 1.0 : y;
}

MaterialCoeffs material_at_xy(const SC x, const SC y, const BatteryMesh & mesh)
{
  const int cell = mesh.cell_index_from_x(x);
  const SC xi = mesh.local_xi_from_x(x);

  if (xi < 0.05)
    return {BatteryRegion::AnodeCollector, 1.0e7, 1.0e-8, 0.0};

  if (xi < 0.40)
  {
    MaterialCoeffs mat{BatteryRegion::Anode, 1.0e4, 1.0, 1.0e6};
    const SC yShift = 0.035 * static_cast<SC>(cell % 5);

    if (in_local_ellipse(xi, y, 0.16, shifted_y(0.25, yShift), 0.035, 0.035) ||
        in_local_ellipse(xi, y, 0.27, shifted_y(0.72, 0.5 * yShift), 0.045, 0.045) ||
        in_local_ellipse(xi, y, 0.34, shifted_y(0.45, 0.8 * yShift), 0.030, 0.030))
    {
      mat.region = BatteryRegion::Inclusion;
      mat.sigma_s = 1.0e7;
      mat.kappa_l = 1.0e-6;
      mat.g = 1.0e2;
    }

    if (in_local_ellipse(xi, y, 0.22, shifted_y(0.52, yShift), 0.040, 0.040) ||
        in_local_ellipse(xi, y, 0.33, shifted_y(0.18, yShift), 0.030, 0.030))
    {
      mat.region = BatteryRegion::Inclusion;
      mat.kappa_l = 1.0e-5;
      mat.g = 1.0e3;
    }

    return mat;
  }

  if (xi < 0.60)
  {
    const int band = static_cast<int>(std::floor(16.0 * y + static_cast<SC>(cell % 3)));
    const bool wetBand = (band % 2 == 0);
    const SC kappa = wetBand ? 1.0e3 : 1.0e-3;
    return {BatteryRegion::Separator, 1.0e-10, kappa, 0.0};
  }

  if (xi < 0.95)
  {
    MaterialCoeffs mat{BatteryRegion::Cathode, 1.0e2, 1.0, 1.0e6};
    const SC yShift = 0.027 * static_cast<SC>((cell + 2) % 7);

    if (in_local_ellipse(xi, y, 0.70, shifted_y(0.32, yShift), 0.050, 0.050) ||
        in_local_ellipse(xi, y, 0.82, shifted_y(0.67, 0.5 * yShift), 0.040, 0.040) ||
        in_local_ellipse(xi, y, 0.90, shifted_y(0.50, 0.8 * yShift), 0.030, 0.030))
    {
      mat.region = BatteryRegion::Inclusion;
      mat.sigma_s = 1.0e6;
      mat.kappa_l = 1.0e-6;
      mat.g = 1.0e2;
    }

    if (in_local_ellipse(xi, y, 0.74, shifted_y(0.78, yShift), 0.035, 0.035) ||
        in_local_ellipse(xi, y, 0.87, shifted_y(0.22, yShift), 0.040, 0.040))
    {
      mat.region = BatteryRegion::Inclusion;
      mat.kappa_l = 1.0e-5;
      mat.g = 1.0e3;
    }

    return mat;
  }

  return {BatteryRegion::CathodeCollector, 1.0e7, 1.0e-8, 0.0};
}

bool is_dirichlet_dof(const BatteryMesh & mesh, const int dof, const GO scalarGid)
{
  const auto [i, j] = mesh.ij_from_scalar_gid(scalarGid);
  (void)j;

  if (dof == 0)
    return i == 0 || i == mesh.nx();

  return scalarGid == mesh.liquid_pin_scalar_gid();
}

bool is_dirichlet_global_gid(const BatteryMesh & mesh, const GO globalGid)
{
  const GO n = mesh.num_scalar_dofs();
  const int dof = globalGid < n ? 0 : 1;
  const GO scalarGid = globalGid < n ? globalGid : globalGid - n;
  return is_dirichlet_dof(mesh, dof, scalarGid);
}

Teuchos::RCP<const map_type>
generate_battery_row_map(const Teuchos::RCP<const comm_type> & comm, const BatteryMesh & mesh)
{
  const GO n = mesh.num_scalar_dofs();
  const GO nTotal = 2 * n;

  auto scalarMap = Teuchos::make_rcp<const map_type>(n, 0, comm);

  std::vector<GO> localGids;
  localGids.reserve(2 * scalarMap->getLocalNumElements());

  for (LO lid = 0; lid < static_cast<LO>(scalarMap->getLocalNumElements()); ++lid)
  {
    const GO scalarGid = scalarMap->getGlobalElement(lid);
    localGids.push_back(scalarGid);
  }

  for (LO lid = 0; lid < static_cast<LO>(scalarMap->getLocalNumElements()); ++lid)
  {
    const GO scalarGid = scalarMap->getGlobalElement(lid);
    localGids.push_back(n + scalarGid);
  }

  return Teuchos::make_rcp<const map_type>(
      nTotal, Teuchos::ArrayView<const GO>(localGids.data(), localGids.size()), 0, comm);
}

Teuchos::RCP<crs_type>
generate_thermal_battery_voltage_matrix(const Teuchos::RCP<const comm_type> & comm,
                                        const BatteryMesh & mesh)
{
  const GO n = mesh.num_scalar_dofs();
  auto rowMap = generate_battery_row_map(comm, mesh);
  auto A = Teuchos::make_rcp<crs_type>(rowMap, 40);

  const SC hx = mesh.hx();
  const SC hy = mesh.hy();

  const SC Kxi[4][4] = {
      {1.0 / 3.0, -1.0 / 3.0, -1.0 / 6.0, 1.0 / 6.0},
      {-1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0, -1.0 / 6.0},
      {-1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0, -1.0 / 3.0},
      {1.0 / 6.0, -1.0 / 6.0, -1.0 / 3.0, 1.0 / 3.0}};

  const SC Keta[4][4] = {
      {1.0 / 3.0, 1.0 / 6.0, -1.0 / 6.0, -1.0 / 3.0},
      {1.0 / 6.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 6.0},
      {-1.0 / 6.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0},
      {-1.0 / 3.0, -1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0}};

  const SC M[4][4] = {
      {4.0, 2.0, 1.0, 2.0},
      {2.0, 4.0, 2.0, 1.0},
      {1.0, 2.0, 4.0, 2.0},
      {2.0, 1.0, 2.0, 4.0}};

  auto insert = [&](const GO row, const GO col, const SC val)
  {
    if (!rowMap->isNodeGlobalElement(row))
      return;

    A->insertGlobalValues(row,
                          Teuchos::ArrayView<const GO>(&col, 1),
                          Teuchos::ArrayView<const SC>(&val, 1));
  };

  auto insert_if_free = [&](const GO row, const GO col, const SC val)
  {
    if (is_dirichlet_global_gid(mesh, row) || is_dirichlet_global_gid(mesh, col))
      return;
    insert(row, col, val);
  };

  for (int ey = 0; ey < mesh.ny; ++ey)
  {
    for (int ex = 0; ex < mesh.nx(); ++ex)
    {
      const SC xc = (static_cast<SC>(ex) + 0.5) * hx;
      const SC yc = (static_cast<SC>(ey) + 0.5) * hy;
      const auto mat = material_at_xy(xc, yc, mesh);

      const GO gid[4] = {
          mesh.node_gid(ex, ey),
          mesh.node_gid(ex + 1, ey),
          mesh.node_gid(ex + 1, ey + 1),
          mesh.node_gid(ex, ey + 1)};

      for (int i = 0; i < 4; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          const SC baseK = (hy / hx) * Kxi[i][j] + (hx / hy) * Keta[i][j];
          const SC Ks = mat.sigma_s * baseK;
          const SC Kl = mat.kappa_l * baseK;
          const SC Mg = mat.g * hx * hy * M[i][j] / 36.0;

          const GO s_i = gid[i];
          const GO s_j = gid[j];
          const GO l_i = n + gid[i];
          const GO l_j = n + gid[j];

          insert_if_free(s_i, s_j, Ks + Mg);
          insert_if_free(s_i, l_j, -Mg);
          insert_if_free(l_i, s_j, -Mg);
          insert_if_free(l_i, l_j, Kl + Mg);
        }
      }
    }
  }

  for (LO lid = 0; lid < static_cast<LO>(rowMap->getLocalNumElements()); ++lid)
  {
    const GO gid = rowMap->getGlobalElement(lid);
    if (is_dirichlet_global_gid(mesh, gid))
    {
      const SC one = 1.0;
      insert(gid, gid, one);
    }
  }

  A->fillComplete(rowMap, rowMap);
  return A;
}

Teuchos::RCP<mv_type>
generate_battery_coordinates(const Teuchos::RCP<const map_type> & map, const BatteryMesh & mesh)
{
  auto coords = Teuchos::make_rcp<mv_type>(map, 2);
  const GO n = mesh.num_scalar_dofs();

  for (LO lid = 0; lid < static_cast<LO>(map->getLocalNumElements()); ++lid)
  {
    const GO gid = map->getGlobalElement(lid);
    const GO scalarGid = gid % n;
    const auto [x, y] = mesh.xy_from_scalar_gid(scalarGid);
    coords->replaceLocalValue(lid, 0, x);
    coords->replaceLocalValue(lid, 1, y);
  }

  return coords;
}

struct AveragedElementMaterial
{
  SC sigma_s{0.0};
  SC kappa_l{0.0};
};

std::vector<AveragedElementMaterial>
build_volume_averaged_element_material(const BatteryMesh & mesh)
{
  constexpr int samplesPerDim = 4;
  constexpr SC invSamples = 1.0 / static_cast<SC>(samplesPerDim * samplesPerDim);

  std::vector<AveragedElementMaterial> elementMaterial(static_cast<std::size_t>(mesh.nx()) * mesh.ny);

  for (int ey = 0; ey < mesh.ny; ++ey)
  {
    for (int ex = 0; ex < mesh.nx(); ++ex)
    {
      SC sigma_s_sum = 0.0;
      SC kappa_l_sum = 0.0;

      for (int sy = 0; sy < samplesPerDim; ++sy)
      {
        const SC y = (static_cast<SC>(ey) + (static_cast<SC>(sy) + 0.5) / samplesPerDim) * mesh.hy();
        for (int sx = 0; sx < samplesPerDim; ++sx)
        {
          const SC x = (static_cast<SC>(ex) + (static_cast<SC>(sx) + 0.5) / samplesPerDim) * mesh.hx();
          const auto mat = material_at_xy(x, y, mesh);
          sigma_s_sum += mat.sigma_s;
          kappa_l_sum += mat.kappa_l;
        }
      }

      elementMaterial[static_cast<std::size_t>(ey) * mesh.nx() + ex] =
          AveragedElementMaterial{sigma_s_sum * invSamples, kappa_l_sum * invSamples};
    }
  }

  return elementMaterial;
}

Teuchos::RCP<mv_type>
generate_battery_volume_averaged_material(const Teuchos::RCP<const map_type> & map,
                                          const BatteryMesh & mesh,
                                          const int dof)
{
  auto material = Teuchos::make_rcp<mv_type>(map, 1);
  const GO n = mesh.num_scalar_dofs();
  const auto elementMaterial = build_volume_averaged_element_material(mesh);

  auto element_value = [&](const int ex, const int ey) -> SC
  {
    const auto & avg = elementMaterial[static_cast<std::size_t>(ey) * mesh.nx() + ex];
    return dof == 0 ? avg.sigma_s : avg.kappa_l;
  };

  for (LO lid = 0; lid < static_cast<LO>(map->getLocalNumElements()); ++lid)
  {
    const GO gid = map->getGlobalElement(lid);
    const GO scalarGid = gid % n;
    const auto [i, j] = mesh.ij_from_scalar_gid(scalarGid);

    SC weightedSum = 0.0;
    SC weightSum = 0.0;
    for (int ey = std::max(0, j - 1); ey <= std::min(mesh.ny - 1, j); ++ey)
    {
      for (int ex = std::max(0, i - 1); ex <= std::min(mesh.nx() - 1, i); ++ex)
      {
        weightedSum += element_value(ex, ey);
        weightSum += 1.0;
      }
    }

    const SC averagedConductivity = weightSum > 0.0 ? weightedSum / weightSum : 0.0;
    material->replaceLocalValue(lid, 0, averagedConductivity);
  }

  return material;
}

double vector_norm2(const vec_type & v)
{
  Teuchos::Array<typename vec_type::mag_type> norms(1);
  v.norm2(norms());
  return static_cast<double>(norms[0]);
}

template<class ST_, class LO_, class GO_, class NO_>
Teuchos::Array<typename Teuchos::ScalarTraits<ST_>::magnitudeType>
explicitResidualNorm2(const Tpetra::Operator<ST_,LO_,GO_,NO_>& A,
                      const Tpetra::MultiVector<ST_,LO_,GO_,NO_>& x,
                      const Tpetra::MultiVector<ST_,LO_,GO_,NO_>& b)
{
  using MV = Tpetra::MultiVector<ST_,LO_,GO_,NO_>;
  MV r(b.getMap(), b.getNumVectors());
  A.apply(x, r);
  r.update(1.0, b, -1.0);
  Teuchos::Array<typename Teuchos::ScalarTraits<ST_>::magnitudeType> n(b.getNumVectors());
  r.norm2(n());
  return n;
}

struct ResidualSummary
{
  double absResidual{0.0};
  double relResidual{0.0};
};

ResidualSummary residual_summary(const Tpetra::Operator<SC,LO,GO,NO> & A,
                                 const mv_type & x,
                                 const mv_type & b)
{
  const auto residuals = explicitResidualNorm2(A, x, b);
  Teuchos::Array<typename mv_type::mag_type> rhsNorms(b.getNumVectors());
  b.norm2(rhsNorms());

  ResidualSummary summary;
  summary.absResidual = static_cast<double>(residuals[0]);
  const double rhsNorm = static_cast<double>(rhsNorms[0]);
  summary.relResidual = summary.absResidual / std::max(rhsNorm, 1.0e-16);
  return summary;
}

Teuchos::RCP<vec_type> extract_single_vector(const Teuchos::RCP<const mv_type> & mv)
{
  auto vec = Teuchos::make_rcp<vec_type>(mv->getMap());
  vec->putScalar(0.0);

  const auto mvView = mv->getLocalViewHost(Tpetra::Access::ReadOnly);
  for (LO lid = 0; lid < static_cast<LO>(mv->getLocalLength()); ++lid)
    vec->replaceLocalValue(lid, mvView(lid, 0));

  return vec;
}

double relative_solution_error(const vec_type & x,
                               const vec_type & xRef)
{
  auto diff = Teuchos::make_rcp<vec_type>(x.getMap());
  diff->update(1.0, x, -1.0, xRef, 0.0);
  return vector_norm2(*diff) / std::max(vector_norm2(xRef), 1.0e-16);
}

double relative_operator_mismatch(const Tpetra::Operator<SC,LO,GO,NO> & A,
                                  const Tpetra::Operator<SC,LO,GO,NO> & B,
                                  const Teuchos::RCP<const map_type> & map,
                                  const BatteryMesh & mesh)
{
  auto probe = Teuchos::make_rcp<mv_type>(map, 1);
  probe->putScalar(1.0);

  for (LO lid = 0; lid < static_cast<LO>(map->getLocalNumElements()); ++lid)
  {
    const GO gid = map->getGlobalElement(lid);
    if (is_dirichlet_global_gid(mesh, gid))
      probe->replaceLocalValue(lid, 0, 0.0);
  }

  auto yA = Teuchos::make_rcp<mv_type>(map, 1);
  auto yB = Teuchos::make_rcp<mv_type>(map, 1);
  auto diff = Teuchos::make_rcp<mv_type>(map, 1);
  A.apply(*probe, *yA);
  B.apply(*probe, *yB);
  diff->update(1.0, *yA, -1.0, *yB, 0.0);

  Teuchos::Array<typename mv_type::mag_type> diffNorm(1), refNorm(1);
  diff->norm2(diffNorm());
  yA->norm2(refNorm());
  return static_cast<double>(diffNorm[0]) / std::max(static_cast<double>(refNorm[0]), 1.0e-16);
}

template<class Comm>
void print_Teuchos_stacked_timer(const Teuchos::RCP<const Comm> & comm)
{
  auto stacked_timer = Teuchos::TimeMonitor::getStackedTimer();
  stacked_timer->stopBaseTimer();

  if (comm->getRank() == 0)
    std::cout << "Linear Solver Timing Summary:\n";

  comm->barrier();

  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = true;
  options.output_minmax = true;
  options.output_histogram = false;
  options.print_warnings = false;

  stacked_timer->report(std::cout, comm, options);

  comm->barrier();

  if (comm->getRank() == 0)
    std::cout << "\n";

  stacked_timer->startBaseTimer();
}

void solveWithAmesos2KLU2(const Teuchos::RCP<const crs_type> & A,
                         const Teuchos::RCP<mv_type> & b,
                         const Teuchos::RCP<mv_type> & x)
{
  auto solver = Amesos2::create<crs_type, mv_type>("KLU2", A, x, b);
  solver->symbolicFactorization();
  solver->numericFactorization();
  solver->solve();
}

} // namespace

int main(int argc, char** argv)
{
  Tpetra::ScopeGuard scope(&argc, &argv);

  auto comm = Tpetra::getDefaultComm();

  int numCells = 10;
  int nxPerCell = 20;
  int ny = 16;
  bool verifyDirect = false;

  std::string solverName = "teko-bgs";
  std::string solverXml  = "";
  int maxIters = 200;
  double tol = 1e-8;

  Teuchos::CommandLineProcessor clp(false, true);
  clp.setOption("num-cells", &numCells, "Number of repeated electrochemical cells in x");
  clp.setOption("nx-per-cell", &nxPerCell, "Elements per cell in the axial direction");
  clp.setOption("ny", &ny, "Elements in the transverse direction");
  clp.setOption("verify-direct", "no-verify-direct", &verifyDirect,
                "Compare iterative solution against Amesos2/KLU2 direct solve");

  clp.setOption("solver", &solverName, "Solver: ifpack2 | teko-bgs | teko-amg");
  clp.setOption("solver-xml", &solverXml,
                "Optional XML file overriding the chosen solver parameter list");
  clp.setOption("max-iters", &maxIters, "GMRES maximum iterations");
  clp.setOption("tol", &tol, "GMRES convergence tolerance");

  auto rc = clp.parse(argc, argv);
  if (rc != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    return (rc == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED ? 0 : 1);

  TEUCHOS_TEST_FOR_EXCEPTION(numCells < 1, std::runtime_error, "num-cells must be positive");
  TEUCHOS_TEST_FOR_EXCEPTION(nxPerCell < 1, std::runtime_error, "nx-per-cell must be positive");
  TEUCHOS_TEST_FOR_EXCEPTION(nxPerCell % 20 != 0, std::runtime_error,
                             "nx-per-cell must be a multiple of 20 to align interfaces");
  TEUCHOS_TEST_FOR_EXCEPTION(ny < 1, std::runtime_error, "ny must be positive");

  multiphys::solvers::SolverChoice which = multiphys::solvers::parseSolverChoice(solverName);

  BatteryMesh mesh{.numCells = numCells, .nxPerCell = nxPerCell, .ny = ny};

  auto A = generate_thermal_battery_voltage_matrix(comm, mesh);
  auto rowMap = A->getRowMap();
  const GO n = mesh.num_scalar_dofs();

  std::vector<std::vector<GO>> blockGids(2);
  for (LO lid = 0; lid < static_cast<LO>(rowMap->getLocalNumElements()); ++lid)
  {
    const GO gid = rowMap->getGlobalElement(lid);
    blockGids[gid < n ? 0 : 1].push_back(gid);
  }

  auto blockedOp = Teuchos::make_rcp<Teko::TpetraHelpers::BlockedTpetraOperator>(blockGids, A);
  if (verifyDirect)
  {
    const bool blockedMatchesFull = blockedOp->testAgainstFullOperator(3, 1.0e-12);
    if (comm->getRank() == 0)
      std::cout << "BlockedTpetraOperator::testAgainstFullOperator="
                << (blockedMatchesFull ? "passed" : "failed") << "\n";
  }

  std::vector<Teuchos::RCP<mv_type>> coords(2);
  std::vector<Teuchos::RCP<mv_type>> material(2);
  for (int block = 0; block < 2; ++block)
  {
    auto Aii = blockedOp->GetBlock(block, block);
    auto map = Aii->getRangeMap();
    coords[block] = generate_battery_coordinates(map, mesh);
    material[block] = generate_battery_volume_averaged_material(map, mesh, block);
  }

  auto x = Teuchos::make_rcp<mv_type>(rowMap, 1);
  auto b = Teuchos::make_rcp<mv_type>(rowMap, 1);
  x->putScalar(0.0);
  b->randomize();

  for (LO lid = 0; lid < static_cast<LO>(rowMap->getLocalNumElements()); ++lid)
  {
    const GO gid = rowMap->getGlobalElement(lid);
    if (is_dirichlet_global_gid(mesh, gid))
      b->replaceLocalValue(lid, 0, 0.0);
  }

  auto bloOp = Teko::createBlockedOp();
  Teko::beginBlockFill(bloOp, 2, 2);
  auto asThyra = [](const auto & op)
  {
    return Thyra::tpetraLinearOp<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(
        Thyra::tpetraVectorSpace<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(op->getRangeMap()),
        Thyra::tpetraVectorSpace<Teko::ST,Teko::LO,Teko::GO,Teko::NT>(op->getDomainMap()),
        op);
  };
  auto A00 = Teuchos::rcp_const_cast<Tpetra::Operator<Teko::ST,Teko::LO,Teko::GO,Teko::NT>>(blockedOp->GetBlock(0, 0));
  auto A01 = Teuchos::rcp_const_cast<Tpetra::Operator<Teko::ST,Teko::LO,Teko::GO,Teko::NT>>(blockedOp->GetBlock(0, 1));
  auto A10 = Teuchos::rcp_const_cast<Tpetra::Operator<Teko::ST,Teko::LO,Teko::GO,Teko::NT>>(blockedOp->GetBlock(1, 0));
  auto A11 = Teuchos::rcp_const_cast<Tpetra::Operator<Teko::ST,Teko::LO,Teko::GO,Teko::NT>>(blockedOp->GetBlock(1, 1));

  Teko::setBlock(0, 0, bloOp, asThyra(A00));
  Teko::setBlock(0, 1, bloOp, asThyra(A01));
  Teko::setBlock(1, 0, bloOp, asThyra(A10));
  Teko::setBlock(1, 1, bloOp, asThyra(A11));
  Teko::endBlockFill(bloOp);

  using blocked_crs_type = Xpetra::BlockedCrsMatrix<SC,LO,GO,NO>;
  auto blockedCrs = Teuchos::make_rcp<blocked_crs_type>(bloOp, Teuchos::null);
  blockedCrs->fillComplete();
  auto monolithicX = blockedCrs->Merge();
  auto AVerify = Xpetra::toTpetra(monolithicX);
  auto AVerifyOp = Teuchos::rcp_dynamic_cast<Tpetra::Operator<SC,LO,GO,NO>>(AVerify, true);

  auto report = [&](const std::string & label,
                    const Tpetra::Operator<SC,LO,GO,NO> & solveOp,
                    const Tpetra::Operator<SC,LO,GO,NO> & reportOp)
  {
    const auto solveRes = residual_summary(solveOp, *x, *b);
    const auto reportRes = residual_summary(reportOp, *x, *b);
    const double relMismatch = std::max(solveRes.relResidual, reportRes.relResidual) > 0.0
        ? std::abs(solveRes.relResidual - reportRes.relResidual)
              / std::max(std::max(solveRes.relResidual, reportRes.relResidual), 1.0e-16)
        : 0.0;
    if (comm->getRank() == 0)
    {
      std::cout << std::scientific << std::setprecision(6)
                << label
                << " solve_abs_residual=" << solveRes.absResidual
                << " solve_rel_residual=" << solveRes.relResidual
                << " abs_residual=" << reportRes.absResidual
                << " rel_residual=" << reportRes.relResidual
                << " residual_mismatch=" << relMismatch
                << " scalar_dofs=" << mesh.num_scalar_dofs()
                << " total_dofs=" << mesh.num_total_dofs()
                << " nnz=" << A->getGlobalNumEntries()
                << "\n";
    }
  };

  auto setUserData = [&](Teuchos::ParameterList & params)
  {
    auto & userData = params.sublist("myInverse").sublist("user data");
    for (int block = 0; block < 2; ++block)
    {
      const auto suffix = std::to_string(block);
      userData.set("Coordinates" + suffix, Xpetra::toXpetra(coords[block]));
      userData.set("Material" + suffix, Xpetra::toXpetra(material[block]));
    }
  };

  switch (which)
  {
    case multiphys::solvers::SolverChoice::Ifpack2SchwarzRILUK:
      multiphys::solvers::solveWithIfpack2SchwarzRILUK_GMRES(A, b, x, maxIters, tol, solverXml);
      report("Ifpack2 + GMRES:", *A, *A);
      break;

    case multiphys::solvers::SolverChoice::TekoBGS:
      multiphys::solvers::solveWithTekoBGS_GMRES(bloOp, b, x, maxIters, tol, solverXml);
      report("Teko BGS + GMRES:", *A, *AVerifyOp);
      break;

    case multiphys::solvers::SolverChoice::TekoMonolithicAMG:
      multiphys::solvers::solveWithTekoMonolithicAMG_GMRES(bloOp, b, x, maxIters, tol, solverXml, setUserData);
      report("Teko Monolithic AMG + GMRES:", *A, *AVerifyOp);
      break;
  }

  if (verifyDirect)
  {
    auto xDirect = Teuchos::make_rcp<mv_type>(rowMap, 1);
    xDirect->putScalar(0.0);
    solveWithAmesos2KLU2(AVerify, b, xDirect);

    const auto iterResidual = residual_summary(*AVerifyOp, *x, *b);
    const auto directResidual = residual_summary(*AVerifyOp, *xDirect, *b);
    const double matrixMismatch = relative_operator_mismatch(*A, *AVerifyOp, rowMap, mesh);

    auto xIterVec = extract_single_vector(Teuchos::rcp_const_cast<const mv_type>(x));
    auto xRefVec = extract_single_vector(Teuchos::rcp_const_cast<const mv_type>(xDirect));
    const double relError = relative_solution_error(*xIterVec, *xRefVec);

    if (comm->getRank() == 0)
    {
      std::cout << std::scientific << std::setprecision(6)
                << "Direct verification: iter_abs_residual=" << iterResidual.absResidual
                << " iter_rel_residual=" << iterResidual.relResidual
                << " direct_abs_residual=" << directResidual.absResidual
                << " direct_rel_residual=" << directResidual.relResidual
                << " rel_error=" << relError
                << " operator_mismatch=" << matrixMismatch << "\n";
    }
  }

  print_Teuchos_stacked_timer(comm);
  return 0;
}
