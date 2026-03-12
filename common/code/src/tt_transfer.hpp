#pragma once

#include <Tpetra_Export.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Teuchos_RCP.hpp>

#include "tt_mesh.hpp"

namespace tt {

using mv_type = Tpetra::MultiVector<SC,LO,GO,NO>;
using export_type = Tpetra::Export<LO,GO,NO>;

// Export overlap monolithic vector to owned monolithic vector.
// Typically:
// - xOwned = Export(INSERT) for exact values (since overlap duplicates are identical)
// - bOwned = Export(ADD)    for RHS (since overlap duplicates contribute)
inline void exportMonolithicVector(const mv_type& xOverlap,
                                   mv_type& xOwned,
                                   const map_type& overlapMonoMap,
                                   const map_type& ownedMonoMap,
                                   Tpetra::CombineMode mode)
{
  export_type exporter(Teuchos::rcpFromRef(overlapMonoMap),
                       Teuchos::rcpFromRef(ownedMonoMap));
  xOwned.putScalar(0.0);
  xOwned.doExport(xOverlap, exporter, mode);
}

} // namespace tt