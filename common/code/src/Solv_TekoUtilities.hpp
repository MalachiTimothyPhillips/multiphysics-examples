/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#ifndef TFTK_TFTK_LINSOLV_LINSOLV_SOLV_TEKO_UTILITIES
#define TFTK_TFTK_LINSOLV_LINSOLV_SOLV_TEKO_UTILITIES

#include "Teko_BlockedTpetraOperator.hpp"
#include "Teko_Utilities.hpp"
#include "Teko_TpetraHelpers.hpp"
#include "Teko_TpetraOperatorWrapper.hpp"
#include "Teko_TpetraThyraConverter.hpp"

namespace experimental
{
using Matrix = Tpetra::CrsMatrix<Teko::ST, Teko::LO, Teko::GO, Teko::NT>;
Teuchos::RCP<const Tpetra::CrsMatrix<Teko::ST, Teko::LO, Teko::GO, Teko::NT>>
get_crs_matrix(int i, int j, const Teko::BlockedLinearOp & blocked);
Teuchos::RCP<const Teuchos::Comm<int>> extract_communicator(const Teko::BlockedLinearOp & blocked);
Teuchos::RCP<Matrix> convert_block_matrix_to_crs_matrix(const Teko::BlockedLinearOp & blocked);
void update_block_matrix_to_crs_matrix(
    const Teko::BlockedLinearOp & blocked, Teuchos::RCP<Matrix> & matrix);
Teko::BlockedLinearOp extract_subblock_matrix(const Teko::BlockedLinearOp & blo,
    const std::vector<int> & rows,
    const std::vector<int> & cols);
Teuchos::RCP<const Teuchos::Comm<int>>
extract_communicator(const Teko::BlockedMultiVector & blocked);
Teko::MultiVector convert_block_vector_to_vector(const Teko::BlockedMultiVector & blocked);
Teuchos::RCP<Tpetra::Map<>> convert_block_range_to_tpetra_range(
    Teuchos::RCP<const Thyra::ProductVectorSpaceBase<Teko::ST>> range,
    Teuchos::RCP<const Teuchos::Comm<int>> comm);

} // end namespace experimental

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
Teuchos::RCP<Thyra::TpetraLinearOp<Scalar, LocalOrdinal, GlobalOrdinal, Node>>
asThyraOp(const Teuchos::RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>> &
        tpetraOperator)
{
  if (!tpetraOperator) return {};
  if (!tpetraOperator->getRangeMap()) return {};

  return Thyra::tpetraLinearOp<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
      Thyra::tpetraVectorSpace<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
          tpetraOperator->getRangeMap()),
      Thyra::tpetraVectorSpace<Scalar, LocalOrdinal, GlobalOrdinal, Node>(
          tpetraOperator->getDomainMap()),
      tpetraOperator);
}

#endif
