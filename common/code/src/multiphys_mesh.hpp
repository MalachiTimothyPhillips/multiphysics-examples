#pragma once

#include <Tpetra_Map.hpp>
#include <Teuchos_Array.hpp>
#include <Kokkos_Core.hpp>

namespace tt {

using SC = double;
using LO = int;
using GO = long long;
using NO = Tpetra::Map<>::node_type;
using map_type = Tpetra::Map<LO,GO,NO>;

using ConnView  = Kokkos::View<GO*[4], Kokkos::HostSpace>;
using CoordView = Kokkos::View<SC*[2], Kokkos::HostSpace>;

inline GO nodeGID(int i, int j, int Nx) { return GO(i) + GO(j)*GO(Nx); }

inline void nodeIJ(GO gid, int Nx, int& i, int& j)
{
  i = int(gid % Nx);
  j = int(gid / Nx);
}

inline bool isBoundaryNode(GO nodeGID, int Nx, int Ny)
{
  int i,j; nodeIJ(nodeGID, Nx, i, j);
  return (i==0 || i==Nx-1 || j==0 || j==Ny-1);
}

inline Teuchos::Array<GO> boundaryNodeGIDs(int Nx, int Ny)
{
  Teuchos::Array<GO> gids;
  gids.reserve(2*Nx + 2*(Ny-2));
  for (int j=0; j<Ny; ++j)
    for (int i=0; i<Nx; ++i) {
      const GO gid = nodeGID(i,j,Nx);
      if (isBoundaryNode(gid, Nx, Ny)) gids.push_back(gid);
    }
  return gids;
}

// Owned elements by "lower-left node owned"
inline ConnView buildOwnedElementConnectivity(const Teuchos::RCP<const map_type>& ownedNodeMap,
                                             int Nx, int Ny)
{
  size_t count = 0;
  for (int j=0; j<Ny-1; ++j)
    for (int i=0; i<Nx-1; ++i)
      if (ownedNodeMap->isNodeGlobalElement(nodeGID(i,j,Nx))) ++count;

  ConnView conn("ownedElemConn", count);

  size_t e = 0;
  for (int j=0; j<Ny-1; ++j) {
    for (int i=0; i<Nx-1; ++i) {
      const GO gLL = nodeGID(i,j,Nx);
      if (!ownedNodeMap->isNodeGlobalElement(gLL)) continue;

      conn(e,0) = gLL;
      conn(e,1) = nodeGID(i+1, j,   Nx);
      conn(e,2) = nodeGID(i+1, j+1, Nx);
      conn(e,3) = nodeGID(i,   j+1, Nx);
      ++e;
    }
  }
  return conn;
}

inline Teuchos::RCP<const map_type>
buildOverlapNodeMap(const Teuchos::RCP<const map_type>& ownedNodeMap,
                    const ConnView& ownedElemConn)
{
  Teuchos::Array<GO> gids;
  gids.reserve(ownedElemConn.extent(0)*4 + ownedNodeMap->getLocalNumElements());

  for (size_t e=0; e<ownedElemConn.extent(0); ++e)
    for (int a=0; a<4; ++a)
      gids.push_back(ownedElemConn(e,a));

  for (LO l=0; l<(LO)ownedNodeMap->getLocalNumElements(); ++l)
    gids.push_back(ownedNodeMap->getGlobalElement(l));

  std::sort(gids.begin(), gids.end());
  gids.erase(std::unique(gids.begin(), gids.end()), gids.end());

  return Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                  gids(), ownedNodeMap->getIndexBase(),
                                  ownedNodeMap->getComm()));
}

inline CoordView buildCoordsStructured(const Teuchos::RCP<const map_type>& nodeMap,
                                      int Nx, int Ny,
                                      SC x0, SC x1, SC y0, SC y1)
{
  (void)Ny;
  const SC dx = (x1-x0)/(Nx-1);
  const SC dy = (y1-y0)/(Ny-1);

  const LO nLocal = (LO)nodeMap->getLocalNumElements();
  CoordView coords("coords", nLocal);

  for (LO l=0; l<nLocal; ++l) {
    const GO gid = nodeMap->getGlobalElement(l);
    int i,j; nodeIJ(gid, Nx, i, j);
    coords(l,0) = x0 + dx*SC(i);
    coords(l,1) = y0 + dy*SC(j);
  }
  return coords;
}

} // namespace tt