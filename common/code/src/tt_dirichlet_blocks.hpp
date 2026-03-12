#pragma once

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <algorithm>

#include "tt_mesh.hpp"
#include "tt_dof.hpp"

namespace tt {

// Set a nodal-block matrix row to I on boundary rows (owned rows only).
template<class SC_, class LO_, class GO_, class NO_>
void applyDirichletRows_DiagBlock(Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& Aii,
                                  Teuchos::ArrayView<GO_> boundaryNodeGIDs)
{
  if (!Aii.isFillComplete())
    throw std::runtime_error("applyDirichletRows_DiagBlock: matrix must be fillComplete()");

  auto rowMap = Aii.getRowMap();
  auto colMap = Aii.getColMap();

  Aii.resumeFill();

  for (GO_ nodeG : boundaryNodeGIDs) {
    const LO_ lrow = rowMap->getLocalElement(nodeG);
    if (lrow == Teuchos::OrdinalTraits<LO_>::invalid()) continue; // not owned

    const size_t nnz = Aii.getNumEntriesInLocalRow(lrow);

    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::nonconst_local_inds_host_view_type
      inds("inds", nnz);
    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::nonconst_values_host_view_type
      vals("vals", nnz);

    size_t n=0;
    Aii.getLocalRowCopy(lrow, inds, vals, n);

    const LO_ ldiag = colMap->getLocalElement(nodeG);
    for (size_t k=0; k<n; ++k)
      vals(k) = (inds(k)==ldiag ? SC_(1) : SC_(0));

    Aii.replaceLocalValues(lrow, n, vals.data(), inds.data());
  }

  Aii.fillComplete(Aii.getDomainMap(), Aii.getRangeMap());
}

// Set a nodal-block matrix row to 0 on boundary rows (owned rows only).
template<class SC_, class LO_, class GO_, class NO_>
void applyDirichletRows_OffDiagBlock(Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& Aij,
                                     Teuchos::ArrayView<GO_> boundaryNodeGIDs)
{
  if (!Aij.isFillComplete())
    throw std::runtime_error("applyDirichletRows_OffDiagBlock: matrix must be fillComplete()");

  auto rowMap = Aij.getRowMap();

  Aij.resumeFill();

  for (GO_ nodeG : boundaryNodeGIDs) {
    const LO_ lrow = rowMap->getLocalElement(nodeG);
    if (lrow == Teuchos::OrdinalTraits<LO_>::invalid()) continue;

    const size_t nnz = Aij.getNumEntriesInLocalRow(lrow);

    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::nonconst_local_inds_host_view_type
      inds("inds", nnz);
    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::nonconst_values_host_view_type
      vals("vals", nnz);

    size_t n=0;
    Aij.getLocalRowCopy(lrow, inds, vals, n);

    for (size_t k=0; k<n; ++k) vals(k) = SC_(0);

    Aij.replaceLocalValues(lrow, n, vals.data(), inds.data());
  }

  Aij.fillComplete(Aij.getDomainMap(), Aij.getRangeMap());
}

// Apply homogeneous Dirichlet to a monolithic RHS vector consistent with blocked dof layout.
// mono dofs: Te=nodeGID, Tl=nodeGID+NglobNodes
template<class SC_, class LO_, class GO_, class NO_>
void applyHomogeneousDirichletToMonolithicRHS2Field(Tpetra::MultiVector<SC_,LO_,GO_,NO_>& b,
                                                    int Nx, int Ny)
{
  const GO_ NglobNodes = GO_(Nx)*GO_(Ny);
  auto map = b.getMap();

  const LO_ nLocal = static_cast<LO_>(map->getLocalNumElements());
  for (LO_ l=0; l<nLocal; ++l) {
    const GO_ g = map->getGlobalElement(l);
    GO_ nodeG = (g >= NglobNodes ? g - NglobNodes : g);
    if (!tt::isBoundaryNode((GO)nodeG, Nx, Ny)) continue;

    for (LO_ j=0; j<(LO_)b.getNumVectors(); ++j)
      b.replaceLocalValue(l, j, SC_(0));
  }
}

} // namespace tt