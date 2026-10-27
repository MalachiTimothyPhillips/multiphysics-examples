/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#pragma once

#include <Tpetra_MultiVector.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayView.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "multiphys_mesh.hpp"

namespace multiphys {

struct FieldExportOptions {
  std::string outputPrefix;
  bool writeCsv{false};
  bool writeVtu{false};
};

struct GatheredStructuredFields {
  struct NodeData {
    GO nodeGid{0};
    int i{0};
    int j{0};
    SC x{0.0};
    SC y{0.0};
    std::vector<SC> fields;
  };

  int Nx{0};
  int Ny{0};
  std::vector<std::string> fieldNames;
  std::vector<NodeData> nodes;
};

namespace detail {

inline std::string csvPath(const std::string & prefix)
{
  return prefix + ".csv";
}

inline std::string vtuPath(const std::string & prefix)
{
  return prefix + ".vtu";
}

inline void validateFieldLayout(const int nFields,
                                const std::vector<std::string> & fieldNames)
{
  TEUCHOS_TEST_FOR_EXCEPTION(nFields <= 0, std::runtime_error,
                             "Field export requires at least one field");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(fieldNames.size()) != nFields,
                             std::runtime_error,
                             "Field export field-name count does not match nFields");
}

inline GatheredStructuredFields gatherStructuredNodalFields(const mv_type & xMono,
                                                            const map_type & ownedNodeMap,
                                                            const int Nx,
                                                            const int Ny,
                                                            const SC x0,
                                                            const SC x1,
                                                            const SC y0,
                                                            const SC y1,
                                                            const int nFields,
                                                            const std::vector<std::string> & fieldNames)
{
  validateFieldLayout(nFields, fieldNames);

  auto comm = ownedNodeMap.getComm();
  const int myRank = comm->getRank();
  const int numRanks = comm->getSize();

  const LO nLocalNodes = static_cast<LO>(ownedNodeMap.getLocalNumElements());
  auto xView = xMono.getLocalViewHost(Tpetra::Access::ReadOnly);

  std::vector<GO> gids(static_cast<std::size_t>(nLocalNodes));
  std::vector<int> ijAndCoords;
  ijAndCoords.reserve(static_cast<std::size_t>(nLocalNodes) * 2);
  std::vector<SC> xy(2 * static_cast<std::size_t>(nLocalNodes), SC(0.0));
  std::vector<SC> fieldValues(static_cast<std::size_t>(nFields) * static_cast<std::size_t>(nLocalNodes), SC(0.0));

  const GO NglobNodes = static_cast<GO>(Nx) * static_cast<GO>(Ny);

  for (LO l = 0; l < nLocalNodes; ++l)
  {
    const GO nodeG = ownedNodeMap.getGlobalElement(l);
    gids[static_cast<std::size_t>(l)] = nodeG;

    int i = 0;
    int j = 0;
    nodeIJ(nodeG, Nx, i, j);

    xy[2 * static_cast<std::size_t>(l)] = x0 + (x1 - x0) * static_cast<SC>(i) / static_cast<SC>(Nx - 1);
    xy[2 * static_cast<std::size_t>(l) + 1] = y0 + (y1 - y0) * static_cast<SC>(j) / static_cast<SC>(Ny - 1);

    for (int f = 0; f < nFields; ++f)
    {
      const GO dofG = nodeG + static_cast<GO>(f) * NglobNodes;
      const LO ldof = xMono.getMap()->getLocalElement(dofG);
      TEUCHOS_TEST_FOR_EXCEPTION(ldof == Teuchos::OrdinalTraits<LO>::invalid(),
                                 std::runtime_error,
                                 "Field export could not find monolithic dof on local map");
      fieldValues[static_cast<std::size_t>(f) * static_cast<std::size_t>(nLocalNodes) + static_cast<std::size_t>(l)] = xView(ldof, 0);
    }
  }

  const int localCount = static_cast<int>(nLocalNodes);
  std::vector<int> counts;
  if (myRank == 0)
    counts.resize(static_cast<std::size_t>(numRanks), 0);

  Teuchos::gatherAll(*comm, 1, &localCount, numRanks, counts.data());

  std::vector<int> displs;
  int totalCount = 0;
  if (myRank == 0)
  {
    displs.resize(static_cast<std::size_t>(numRanks), 0);
    for (int r = 0; r < numRanks; ++r)
    {
      displs[static_cast<std::size_t>(r)] = totalCount;
      totalCount += counts[static_cast<std::size_t>(r)];
    }
  }

  std::vector<GO> gatheredGids;
  std::vector<SC> gatheredXY;
  std::vector<SC> gatheredFields;
  if (myRank == 0)
  {
    gatheredGids.resize(static_cast<std::size_t>(totalCount));
    gatheredXY.resize(2 * static_cast<std::size_t>(totalCount), SC(0.0));
    gatheredFields.resize(static_cast<std::size_t>(nFields) * static_cast<std::size_t>(totalCount), SC(0.0));
  }

  auto gatherVector = [&](auto * sendBuf, auto * recvBuf, const int stride)
  {
    using value_type = std::remove_pointer_t<decltype(sendBuf)>;
    for (int r = 0; r < numRanks; ++r)
    {
      const int recvCount = (myRank == 0) ? counts[static_cast<std::size_t>(r)] : 0;
      std::vector<value_type> tmp;
      if (myRank == r)
      {
        tmp.assign(sendBuf, sendBuf + static_cast<std::size_t>(stride * localCount));
      }
      if (myRank == 0)
      {
        std::vector<value_type> recvTmp(static_cast<std::size_t>(stride * recvCount));
        if (r == 0)
        {
          std::copy(tmp.begin(), tmp.end(), recvTmp.begin());
        }
        else
        {
          Teuchos::receive(*comm, recvCount * stride, r, recvTmp.data());
        }
        std::copy(recvTmp.begin(), recvTmp.end(), recvBuf + static_cast<std::size_t>(stride * displs[static_cast<std::size_t>(r)]));
      }
      else if (myRank == r)
      {
        Teuchos::send(*comm, stride * localCount, tmp.data(), 0);
      }
    }
  };

  gatherVector(gids.data(), gatheredGids.data(), 1);
  gatherVector(xy.data(), gatheredXY.data(), 2);
  for (int f = 0; f < nFields; ++f)
  {
    gatherVector(fieldValues.data() + static_cast<std::size_t>(f) * static_cast<std::size_t>(localCount),
                 gatheredFields.data() + static_cast<std::size_t>(f) * static_cast<std::size_t>(totalCount),
                 1);
  }

  GatheredStructuredFields out;
  if (myRank != 0)
    return out;

  out.Nx = Nx;
  out.Ny = Ny;
  out.fieldNames = fieldNames;
  out.nodes.reserve(static_cast<std::size_t>(totalCount));

  for (int idx = 0; idx < totalCount; ++idx)
  {
    const GO nodeG = gatheredGids[static_cast<std::size_t>(idx)];
    int i = 0;
    int j = 0;
    nodeIJ(nodeG, Nx, i, j);

    typename GatheredStructuredFields::NodeData node;
    node.nodeGid = nodeG;
    node.i = i;
    node.j = j;
    node.x = gatheredXY[2 * static_cast<std::size_t>(idx)];
    node.y = gatheredXY[2 * static_cast<std::size_t>(idx) + 1];
    node.fields.resize(static_cast<std::size_t>(nFields), SC(0.0));
    for (int f = 0; f < nFields; ++f)
      node.fields[static_cast<std::size_t>(f)] = gatheredFields[static_cast<std::size_t>(f) * static_cast<std::size_t>(totalCount) + static_cast<std::size_t>(idx)];

    out.nodes.push_back(std::move(node));
  }

  std::sort(out.nodes.begin(), out.nodes.end(), [](const auto & a, const auto & b) {
    return std::tie(a.j, a.i) < std::tie(b.j, b.i);
  });

  return out;
}

inline void writeStructuredCsv(const GatheredStructuredFields & data,
                               const std::string & path,
                               const bool addDisplacementMagnitude)
{
  std::ofstream out(path);
  TEUCHOS_TEST_FOR_EXCEPTION(!out, std::runtime_error, "Could not open CSV output file: " + path);

  out << std::setprecision(16);
  out << "node_gid,i,j,x,y";
  for (const auto & name : data.fieldNames)
    out << "," << name;
  if (addDisplacementMagnitude)
    out << ",u_mag";
  out << "\n";

  for (const auto & node : data.nodes)
  {
    out << node.nodeGid << ","
        << node.i << ","
        << node.j << ","
        << node.x << ","
        << node.y;
    for (const auto value : node.fields)
      out << "," << value;
    if (addDisplacementMagnitude)
    {
      const SC ux = node.fields.size() > 0 ? node.fields[0] : SC(0.0);
      const SC uy = node.fields.size() > 1 ? node.fields[1] : SC(0.0);
      out << "," << std::sqrt(ux * ux + uy * uy);
    }
    out << "\n";
  }
}

inline void writeStructuredVtu(const GatheredStructuredFields & data,
                               const std::string & path,
                               const bool includeDisplacementVector,
                               const int displacementField0,
                               const int displacementField1,
                               const int scalarFieldForMagnitude)
{
  std::ofstream out(path);
  TEUCHOS_TEST_FOR_EXCEPTION(!out, std::runtime_error, "Could not open VTU output file: " + path);

  const int Nx = data.Nx;
  const int Ny = data.Ny;
  const int nPoints = static_cast<int>(data.nodes.size());
  const int nCells = (Nx - 1) * (Ny - 1);

  TEUCHOS_TEST_FOR_EXCEPTION(nPoints != Nx * Ny, std::runtime_error,
                             "VTU writer expected Nx*Ny gathered nodes");

  out << std::setprecision(16);
  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <UnstructuredGrid>\n";
  out << "    <Piece NumberOfPoints=\"" << nPoints << "\" NumberOfCells=\"" << nCells << "\">\n";
  out << "      <PointData>\n";

  for (std::size_t f = 0; f < data.fieldNames.size(); ++f)
  {
    out << "        <DataArray type=\"Float64\" Name=\"" << data.fieldNames[f]
        << "\" NumberOfComponents=\"1\" format=\"ascii\">\n          ";
    for (int p = 0; p < nPoints; ++p)
      out << data.nodes[static_cast<std::size_t>(p)].fields[f] << " ";
    out << "\n        </DataArray>\n";
  }

  if (includeDisplacementVector)
  {
    out << "        <DataArray type=\"Float64\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
    for (int p = 0; p < nPoints; ++p)
    {
      const auto & fields = data.nodes[static_cast<std::size_t>(p)].fields;
      const SC ux = fields[static_cast<std::size_t>(displacementField0)];
      const SC uy = fields[static_cast<std::size_t>(displacementField1)];
      out << ux << " " << uy << " 0 ";
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"Float64\" Name=\"u_mag\" NumberOfComponents=\"1\" format=\"ascii\">\n          ";
    for (int p = 0; p < nPoints; ++p)
    {
      const auto & fields = data.nodes[static_cast<std::size_t>(p)].fields;
      const SC ux = fields[static_cast<std::size_t>(displacementField0)];
      const SC uy = fields[static_cast<std::size_t>(displacementField1)];
      out << std::sqrt(ux * ux + uy * uy) << " ";
    }
    out << "\n        </DataArray>\n";
  }

  (void)scalarFieldForMagnitude;

  out << "      </PointData>\n";
  out << "      <CellData/>\n";
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
  for (const auto & node : data.nodes)
    out << node.x << " " << node.y << " 0 ";
  out << "\n        </DataArray>\n";
  out << "      </Points>\n";

  out << "      <Cells>\n";
  out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n          ";
  for (int j = 0; j < Ny - 1; ++j)
  {
    for (int i = 0; i < Nx - 1; ++i)
    {
      const int n0 = j * Nx + i;
      const int n1 = n0 + 1;
      const int n2 = n0 + Nx + 1;
      const int n3 = n0 + Nx;
      out << n0 << " " << n1 << " " << n2 << " " << n3 << " ";
    }
  }
  out << "\n        </DataArray>\n";

  out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n          ";
  for (int c = 1; c <= nCells; ++c)
    out << 4 * c << " ";
  out << "\n        </DataArray>\n";

  out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          ";
  for (int c = 0; c < nCells; ++c)
    out << 9 << " ";
  out << "\n        </DataArray>\n";
  out << "      </Cells>\n";

  out << "    </Piece>\n";
  out << "  </UnstructuredGrid>\n";
  out << "</VTKFile>\n";
}

} // namespace detail

inline void exportStructuredMonolithicFields(const mv_type & xMono,
                                             const map_type & ownedNodeMap,
                                             const int Nx,
                                             const int Ny,
                                             const SC x0,
                                             const SC x1,
                                             const SC y0,
                                             const SC y1,
                                             const int nFields,
                                             const std::vector<std::string> & fieldNames,
                                             const FieldExportOptions & options,
                                             const bool addDisplacementMagnitude = false,
                                             const bool includeDisplacementVector = false,
                                             const int displacementField0 = 0,
                                             const int displacementField1 = 1)
{
  if (!options.writeCsv && !options.writeVtu)
    return;

  TEUCHOS_TEST_FOR_EXCEPTION(options.outputPrefix.empty(), std::runtime_error,
                             "Field export requested, but output prefix is empty");

  const auto gathered = detail::gatherStructuredNodalFields(xMono, ownedNodeMap,
                                                            Nx, Ny, x0, x1, y0, y1,
                                                            nFields, fieldNames);

  auto comm = ownedNodeMap.getComm();
  if (comm->getRank() != 0)
    return;

  if (options.writeCsv)
    detail::writeStructuredCsv(gathered, detail::csvPath(options.outputPrefix), addDisplacementMagnitude);

  if (options.writeVtu)
    detail::writeStructuredVtu(gathered, detail::vtuPath(options.outputPrefix),
                               includeDisplacementVector,
                               displacementField0,
                               displacementField1,
                               addDisplacementMagnitude ? displacementField0 : -1);
}

} // namespace multiphys
