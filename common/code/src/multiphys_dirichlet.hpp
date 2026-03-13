/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#pragma once

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <vector>
#include <algorithm>

#include "multiphys_mesh.hpp"

namespace tt {

template<class SC_, class LO_, class GO_, class NO_>
void applyHomogeneousDirichlet_Monolithic2Field(
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& A,
    Tpetra::MultiVector<SC_,LO_,GO_,NO_>* b, // nullable
    int Nx, int Ny)
{
  if (!A.isFillComplete()) throw std::runtime_error("Dirichlet: A must be fillComplete()");
  const GO_ NglobNodes = GO_(Nx)*GO_(Ny);

  auto rowMap = A.getRowMap();
  auto colMap = A.getColMap();

  A.resumeFill();

  const LO_ nLocalRows = (LO_)rowMap->getLocalNumElements();
  for (LO_ lrow=0; lrow<nLocalRows; ++lrow) {
    const GO_ grow = rowMap->getGlobalElement(lrow);
    GO_ nodeG = (grow >= NglobNodes ? grow - NglobNodes : grow);

    if (!tt::isBoundaryNode((GO)nodeG, Nx, Ny)) continue;

    const size_t nnz = A.getNumEntriesInLocalRow(lrow);

    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::nonconst_local_inds_host_view_type
      inds("inds", nnz);
    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::nonconst_values_host_view_type
      vals("vals", nnz);

    size_t n=0;
    A.getLocalRowCopy(lrow, inds, vals, n);

    const LO_ ldiag = colMap->getLocalElement(grow);
    for (size_t k=0; k<n; ++k) vals(k) = (inds(k)==ldiag ? SC_(1) : SC_(0));
    A.replaceLocalValues(lrow, n, vals.data(), inds.data());

    if (b) {
      for (LO_ j=0; j<(LO_)b->getNumVectors(); ++j)
        b->replaceLocalValue(lrow, j, SC_(0));
    }
  }

  A.fillComplete(A.getDomainMap(), A.getRangeMap());
}

// Apply homogeneous Dirichlet to a monolithic nodal system with N fields.
//
// Assumptions:
//  - There are N scalar dofs per node.
//  - Global dof numbering is field-blocked by node GID offsets:
//      field 0: nodeGID + 0*NglobNodes
//      field 1: nodeGID + 1*NglobNodes
//      ...
//      field (N-1): nodeGID + (N-1)*NglobNodes
//  - Dirichlet is applied on ALL fields at boundary nodes.
//  - A is fillComplete() on entry.
//
// Effect:
//  - For owned rows corresponding to boundary nodes, replace row with identity row
//    (1 on diagonal, 0 elsewhere) and set RHS entry to 0 (if b != nullptr).
template<class SC_, class LO_, class GO_, class NO_>
void applyHomogeneousDirichlet_MonolithicNField(
    Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>& A,
    Tpetra::MultiVector<SC_,LO_,GO_,NO_>* b, // nullable
    int Nx, int Ny,
    int nFields)
{
  if (nFields <= 0) throw std::runtime_error("Dirichlet: nFields must be positive");
  if (!A.isFillComplete()) throw std::runtime_error("Dirichlet: A must be fillComplete()");

  const GO_ NglobNodes = GO_(Nx) * GO_(Ny);

  auto rowMap = A.getRowMap();
  auto colMap = A.getColMap();

  A.resumeFill();

  const LO_ nLocalRows = static_cast<LO_>(rowMap->getLocalNumElements());
  for (LO_ lrow = 0; lrow < nLocalRows; ++lrow) {
    const GO_ grow = rowMap->getGlobalElement(lrow);

    // Map monolithic dof GID -> node GID via modulo arithmetic
    // (works with GO not necessarily nonnegative as long as numbering is standard nonnegative).
    const GO_ nodeG = grow % NglobNodes;

    if (!tt::isBoundaryNode(static_cast<GO>(nodeG), Nx, Ny)) continue;

    const size_t nnz = A.getNumEntriesInLocalRow(lrow);

    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::nonconst_local_inds_host_view_type
      inds("inds", nnz);
    typename Tpetra::CrsMatrix<SC_,LO_,GO_,NO_>::nonconst_values_host_view_type
      vals("vals", nnz);

    size_t n = 0;
    A.getLocalRowCopy(lrow, inds, vals, n);

    const LO_ ldiag = colMap->getLocalElement(grow);
    for (size_t k = 0; k < n; ++k)
      vals(k) = (inds(k) == ldiag ? SC_(1) : SC_(0));

    A.replaceLocalValues(lrow, n, vals.data(), inds.data());

    if (b) {
      for (LO_ j = 0; j < static_cast<LO_>(b->getNumVectors()); ++j)
        b->replaceLocalValue(lrow, j, SC_(0));
    }
  }

  A.fillComplete(A.getDomainMap(), A.getRangeMap());
}

} // namespace tt