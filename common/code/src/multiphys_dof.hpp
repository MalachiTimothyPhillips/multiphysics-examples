#pragma once

#include <Tpetra_Map.hpp>
#include <Teuchos_RCP.hpp>

#include "multiphys_mesh.hpp"

namespace tt {

// Build blocked monolithic map from a nodal map:
// mono = [ nodeGID ] union [ nodeGID + NglobNodes ] with the same ownership/ghosting as nodeMap.
inline Teuchos::RCP<const map_type>
buildMonolithicMap2FieldFromNodeMap(const Teuchos::RCP<const map_type>& nodeMap,
                                    GO NglobNodes)
{
  const LO nLocal = static_cast<LO>(nodeMap->getLocalNumElements());

  Teuchos::Array<GO> monoGIDs;
  monoGIDs.reserve(2 * nLocal);

  // Preserve local order: first field then second field
  for (LO l = 0; l < nLocal; ++l) {
    const GO g = nodeMap->getGlobalElement(l);
    monoGIDs.push_back(g);
  }
  for (LO l = 0; l < nLocal; ++l) {
    const GO g = nodeMap->getGlobalElement(l);
    monoGIDs.push_back(g + NglobNodes);
  }

  return Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                  monoGIDs(),
                                  nodeMap->getIndexBase(),
                                  nodeMap->getComm()));
}

// mono dofs (n fields): u1=nodeGID, u2=nodeGID+N, ..., un=nodeGID+(n-1)N
inline Teuchos::RCP<const map_type>
buildMonolithicMapNFieldFromNodeMap(const Teuchos::RCP<const map_type>& nodeMap,
                                    GO NglobNodes,
                                    LO numFields)
{
  const LO nLocal = static_cast<LO>(nodeMap->getLocalNumElements());

  Teuchos::Array<GO> monoGIDs;
  monoGIDs.reserve(numFields * nLocal);

  GO fieldOffset = 0;
  for(LO fld = 0; fld < numFields; ++fld) {
    for (LO l = 0; l < nLocal; ++l) {
      monoGIDs.push_back(nodeMap->getGlobalElement(l) + fieldOffset);
    }

    fieldOffset += NglobNodes;
  }

  return Teuchos::rcp(new map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                  monoGIDs(),
                                  nodeMap->getIndexBase(),
                                  nodeMap->getComm()));
}

} // namespace tt