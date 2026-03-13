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
#include <Teuchos_RCP.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include "multiphys_mesh.hpp"

namespace tt {

// Build 2D elasticity rigid body modes (RBMs) as a Tpetra::MultiVector with 3 columns.
// Ordering in uMap (2-field monolithic): ux(nodeG), uy(nodeG + NglobNodes).
//
// Inputs:
//  - nodeMap: nodal map (owned or overlap) used to define coordinates
//  - uMap: 2-field monolithic map built from nodeMap (same ownership/order assumptions)
//  - coords: coordinates on nodeMap (local length = nodeMap local length)
//  - NglobNodes: global number of nodes (Nx*Ny for your structured grid)
//  - x0,y0: rotation origin (use domain center or 0,0)
//
// Output:
//  - ns: MultiVector(uMap, 3) with columns [Tx, Ty, Rz]
inline Teuchos::RCP<Tpetra::MultiVector<SC,LO,GO,NO>>
buildRigidBodyModes2D(const Teuchos::RCP<const map_type>& nodeMap,
                      const Teuchos::RCP<const map_type>& uMap,
                      const CoordView& coords,
                      GO NglobNodes,
                      SC x0, SC y0)
{
  using MV = Tpetra::MultiVector<SC,LO,GO,NO>;

  TEUCHOS_TEST_FOR_EXCEPTION(nodeMap.is_null() || uMap.is_null(),
                             std::runtime_error, "buildRigidBodyModes2D: null map");
  TEUCHOS_TEST_FOR_EXCEPTION(coords.extent(0) != nodeMap->getLocalNumElements(),
                             std::runtime_error, "buildRigidBodyModes2D: coords/nodeMap mismatch");

  // 3 rigid body modes in 2D
  auto ns = Teuchos::rcp(new MV(uMap, 3));
  ns->putScalar(0.0);

  const LO nLocalNodes = (LO)nodeMap->getLocalNumElements();

  auto nsView = ns->getLocalViewHost(Tpetra::Access::OverwriteAll);

  for (LO l=0; l<nLocalNodes; ++l) {
    const GO nodeG = nodeMap->getGlobalElement(l);

    // Local row indices in the 2-field monolithic layout:
    // uMap local ordering is assumed to be [all nodeGIDs][all nodeGIDs + NglobNodes]
    const LO lUx = uMap->getLocalElement(nodeG);
    const LO lUy = uMap->getLocalElement(nodeG + NglobNodes);

    TEUCHOS_TEST_FOR_EXCEPTION(lUx == Teuchos::OrdinalTraits<LO>::invalid() ||
                               lUy == Teuchos::OrdinalTraits<LO>::invalid(),
                               std::runtime_error, "buildRigidBodyModes2D: uMap missing expected dof");

    const SC x = coords(l,0);
    const SC y = coords(l,1);

    // Mode 0: translation in x
    nsView(lUx, 0) = 1.0;
    nsView(lUy, 0) = 0.0;

    // Mode 1: translation in y
    nsView(lUx, 1) = 0.0;
    nsView(lUy, 1) = 1.0;

    // Mode 2: rotation about (x0,y0): u = [- (y-y0), (x-x0)]
    nsView(lUx, 2) = -(y - y0);
    nsView(lUy, 2) =  (x - x0);
  }

  return ns;
}

} // namespace tt