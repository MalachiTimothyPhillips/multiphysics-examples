/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#include "Solv_TekoUtilities.hpp"
#include "Tpetra_Details_makeColMap_decl.hpp"
#include "KokkosSparse_SortCrs.hpp"
#include <stack>

namespace experimental
{

Teuchos::RCP<const Tpetra::CrsMatrix<Teko::ST, Teko::LO, Teko::GO, Teko::NT>>
get_crs_matrix(int i, int j, const Teko::BlockedLinearOp & blocked)
{
  bool transposed = false;
  Teko::ST scalar = 0.0;
  return Teko::TpetraHelpers::getTpetraCrsMatrix(
      Teko::getBlock(i, j, blocked), &scalar, &transposed);
}

namespace
{

auto compute_block_offsets(const Teko::BlockedLinearOp & blocked)
{
  using Teko::GO;
  using Teko::LO;
  using Teko::NT;
  using Teko::ST;
  using device_type = typename NT::device_type;
  const int nBlockRows = Teko::blockRowCount(blocked);

  size_t numLocalRows = 0;
  GO numGlobalRows = 0;
  std::vector<GO> scanLocalBlockOffset(nBlockRows + 1, 0);
  std::vector<GO> scanGlobalBlockOffset(nBlockRows + 1, 0);

  for (int row = 0; row < nBlockRows; ++row)
  {
    auto A_ii = get_crs_matrix(row, row, blocked);

    auto rowMap_ii = A_ii->getRowMap();
    numGlobalRows += rowMap_ii->getGlobalNumElements();
    numLocalRows += rowMap_ii->getLocalNumElements();

    scanLocalBlockOffset[row + 1] = numLocalRows;
    scanGlobalBlockOffset[row + 1] = numGlobalRows;
  }

  Kokkos::View<GO *, device_type> scanLocalBlockOffsetView(
      Kokkos::ViewAllocateWithoutInitializing("scanLocalBlockOffsetView"), nBlockRows + 1);
  {
    auto hostMirror = Kokkos::create_mirror_view(scanLocalBlockOffsetView);
    std::copy(scanLocalBlockOffset.begin(), scanLocalBlockOffset.end(), hostMirror.data());
    Kokkos::deep_copy(scanLocalBlockOffsetView, hostMirror);
  }

  Kokkos::View<GO *, device_type> scanGlobalBlockOffsetView(
      Kokkos::ViewAllocateWithoutInitializing("scanLocalBlockOffsetView"), nBlockRows + 1);
  {
    auto hostMirror = Kokkos::create_mirror_view(scanGlobalBlockOffsetView);
    std::copy(scanGlobalBlockOffset.begin(), scanGlobalBlockOffset.end(), hostMirror.data());
    Kokkos::deep_copy(scanGlobalBlockOffsetView, hostMirror);
  }

  return std::make_tuple(
      numLocalRows, numGlobalRows, scanLocalBlockOffsetView, scanGlobalBlockOffsetView);
}

auto extract_view_of_local_subblock_matrices_and_maps(const Teko::BlockedLinearOp & blocked)
{
  using Teko::GO;
  using Teko::LO;
  using Teko::NT;
  using Teko::ST;
  using local_matrix_type = Tpetra::CrsMatrix<ST, LO, GO, NT>::local_matrix_device_type;
  using local_map_type = Tpetra::Map<LO, GO, NT>::local_map_type;

  const int nBlockRows = Teko::blockRowCount(blocked);
  const int nBlockCols = Teko::blockColCount(blocked);
  Kokkos::View<local_matrix_type **, Kokkos::SharedSpace> subblockMatrices(
      Kokkos::view_alloc("subblockMatrices", Kokkos::SequentialHostInit), nBlockRows, nBlockRows);
  Kokkos::View<local_map_type **, Kokkos::SharedSpace> subblockColMaps(
      Kokkos::view_alloc("subblockColMap", Kokkos::SequentialHostInit), nBlockRows, nBlockRows);
  for (int row = 0; row < nBlockRows; ++row)
  {
    for (int col = 0; col < nBlockCols; ++col)
    {
      auto A_ij = get_crs_matrix(row, col, blocked);
      subblockMatrices(row, col) = local_matrix_type(A_ij->getLocalMatrixDevice());
      subblockColMaps(row, col) = local_map_type(A_ij->getColMap()->getLocalMap());
    }
  }
  return std::make_tuple(subblockMatrices, subblockColMaps);
}

template <typename OffsetViewType>
KOKKOS_INLINE_FUNCTION int
compute_block_row(const OffsetViewType & scanOffsetView, Teko::LO localRow)
{
  int blockRow = -1;
  const auto nBlockRows = scanOffsetView.extent(0) - 1;
  for (auto row = 0U; row < nBlockRows; ++row)
  {
    const auto lower = scanOffsetView(row);
    const auto upper = scanOffsetView(row + 1);
    if (lower <= localRow && localRow < upper)
    {
      blockRow = row;
      return blockRow;
    }
  }
  return blockRow;
}

} // namespace

Teuchos::RCP<const Teuchos::Comm<int>> extract_communicator(const Teko::BlockedLinearOp & blocked)
{
  auto A_ij = get_crs_matrix(0, 0, blocked);
  auto rowMap_ij = A_ij->getRowMap();
  return rowMap_ij->getComm();
}

Teuchos::RCP<Matrix> convert_block_matrix_to_crs_matrix(const Teko::BlockedLinearOp & blocked)
{
  using Teko::GO;
  using Teko::LO;
  using Teko::NT;
  using Teko::ST;

  using local_matrix_type = Tpetra::CrsMatrix<ST, LO, GO, NT>::local_matrix_device_type;
  using local_map_type = Tpetra::Map<LO, GO, NT>::local_map_type;
  using row_map_type = local_matrix_type::row_map_type::non_const_type;
  using values_type = local_matrix_type::values_type::non_const_type;
  using index_type = local_matrix_type::index_type::non_const_type;
  using matrix_execution_space =
      typename Tpetra::CrsMatrix<ST, LO, GO, NT>::local_matrix_device_type::execution_space;
  using device_type = typename NT::device_type;

  const int nBlockRows = Teko::blockRowCount(blocked);
  const int nBlockCols = Teko::blockColCount(blocked);

  auto tComm = extract_communicator(blocked);

  GO numLocalRows{0};
  GO numGlobalRows{0};
  Kokkos::View<GO *, device_type> scanLocalBlockOffsetView;
  Kokkos::View<GO *, device_type> scanGlobalBlockOffsetView;

  std::tie(numLocalRows, numGlobalRows, scanLocalBlockOffsetView, scanGlobalBlockOffsetView) =
      compute_block_offsets(blocked);

  const auto invalidLO = Teuchos::OrdinalTraits<LO>::invalid();
  const int numRows = numLocalRows;
  auto nEntriesPerRow = Kokkos::View<LO *, device_type>(
      Kokkos::ViewAllocateWithoutInitializing("nEntriesPerRow"), numRows);
  Kokkos::deep_copy(nEntriesPerRow, invalidLO);
  auto prefixSumEntriesPerRow =
      row_map_type(Kokkos::ViewAllocateWithoutInitializing("prefixSumEntriesPerRow"), numRows + 1);

  Kokkos::View<local_matrix_type **, Kokkos::SharedSpace> subblockMatrices;
  Kokkos::View<local_map_type **, Kokkos::SharedSpace> subblockColMaps;
  std::tie(subblockMatrices, subblockColMaps) =
      extract_view_of_local_subblock_matrices_and_maps(blocked);

  LO totalNumOwnedCols = 0;
  Kokkos::parallel_scan(
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, matrix_execution_space>(0, numRows),
      KOKKOS_LAMBDA(const LO localRow, LO & sumNumEntries, bool finalPass) {
        auto numOwnedCols = nEntriesPerRow(localRow);

        if (numOwnedCols == invalidLO)
        {
          const auto blockRow = compute_block_row(scanLocalBlockOffsetView, localRow);
          numOwnedCols = 0;
          const auto lid = localRow - scanLocalBlockOffsetView(blockRow);
          for (int col = 0; col < nBlockCols; ++col)
          {
            numOwnedCols += subblockMatrices(blockRow, col).row(lid).length;
          }
          nEntriesPerRow(localRow) = numOwnedCols;
        }

        if (finalPass)
        {
          prefixSumEntriesPerRow(localRow) = sumNumEntries;
          if (localRow == (numRows - 1))
          {
            prefixSumEntriesPerRow(numRows) = prefixSumEntriesPerRow(localRow) + numOwnedCols;
          }
        }
        sumNumEntries += numOwnedCols;
      },
      totalNumOwnedCols);

  auto columnIndices = Kokkos::View<GO *, device_type>(
      Kokkos::ViewAllocateWithoutInitializing("columnIndices"), totalNumOwnedCols);
  auto values = values_type(Kokkos::ViewAllocateWithoutInitializing("values"), totalNumOwnedCols);

  LO maxNumEntriesSubblock = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, matrix_execution_space>(0, numRows),
      KOKKOS_LAMBDA(const LO localRow, LO & maxNumEntries) {
        const auto blockRow = compute_block_row(scanLocalBlockOffsetView, localRow);

        LO colId = 0;
        LO colIdStart = prefixSumEntriesPerRow[localRow];
        const auto lid = localRow - scanLocalBlockOffsetView(blockRow);
        for (int blockCol = 0; blockCol < nBlockCols; ++blockCol)
        {
          auto col_map = subblockColMaps(blockRow, blockCol);
          const auto gidStart = scanGlobalBlockOffsetView(blockCol);
          const auto sparseRowView = subblockMatrices(blockRow, blockCol).row(lid);
          for (auto col = 0; col < sparseRowView.length; col++)
          {
            auto colidx = col_map.getGlobalElement(sparseRowView.colidx(col));
            auto value = sparseRowView.value(col);
            values[colId + colIdStart] = value;
            columnIndices[colId + colIdStart] = colidx + gidStart;
            colId++;
          }
        }

        const auto numOwnedCols = nEntriesPerRow(localRow);
        maxNumEntries = Kokkos::max(maxNumEntries, numOwnedCols);
      },
      Kokkos::Max<LO>(maxNumEntriesSubblock));

  Teko::LinearOp blockedOp_lo = blocked;
  auto blockedOp_tpetra =
      Teuchos::rcp(new Teko::TpetraHelpers::TpetraOperatorWrapper(blockedOp_lo));
  auto rangeMap = blockedOp_tpetra->getRangeMap();
  auto domainMap = blockedOp_tpetra->getDomainMap();

  Teuchos::RCP<const Tpetra::Map<LO, GO, NT>> colMap;
  Tpetra::Details::makeColMap<LO, GO, NT>(colMap, domainMap, columnIndices);
  TEUCHOS_ASSERT(colMap);

  auto colMap_dev = colMap->getLocalMap();
  auto localColumnIndices =
      index_type(Kokkos::ViewAllocateWithoutInitializing("localColumnIndices"), totalNumOwnedCols);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, matrix_execution_space>(
          0, totalNumOwnedCols),
      KOKKOS_LAMBDA(const LO index) {
        localColumnIndices(index) = colMap_dev.getLocalElement(columnIndices(index));
      });

  KokkosSparse::sort_crs_matrix<matrix_execution_space, row_map_type, index_type, values_type>(
      prefixSumEntriesPerRow, localColumnIndices, values);

  auto lcl_mat = Tpetra::CrsMatrix<ST, LO, GO, NT>::local_matrix_device_type("localMat",
      numRows,
      maxNumEntriesSubblock,
      totalNumOwnedCols,
      values,
      prefixSumEntriesPerRow,
      localColumnIndices);

  Teuchos::RCP<Tpetra::CrsMatrix<ST, LO, GO, NT>> mat =
      rcp(new Tpetra::CrsMatrix<ST, LO, GO, NT>(lcl_mat, rangeMap, colMap, domainMap, rangeMap));

  return mat;
}

void update_block_matrix_to_crs_matrix(
    const Teko::BlockedLinearOp & blocked, Teuchos::RCP<Matrix> & matrix)
{
  using Teko::GO;
  using Teko::LO;
  using Teko::NT;
  using Teko::ST;
  using device_type = typename NT::device_type;
  using local_matrix_type = Tpetra::CrsMatrix<ST, LO, GO, NT>::local_matrix_device_type;
  using local_map_type = Tpetra::Map<LO, GO, NT>::local_map_type;
  using matrix_execution_space =
      typename Tpetra::CrsMatrix<ST, LO, GO, NT>::local_matrix_device_type::execution_space;

  const int nBlockRows = Teko::blockRowCount(blocked);
  const int nBlockCols = Teko::blockColCount(blocked);

  GO numLocalRows{0};
  GO numGlobalRows{0};
  Kokkos::View<GO *, device_type> scanLocalBlockOffsetView;
  Kokkos::View<GO *, device_type> scanGlobalBlockOffsetView;

  std::tie(numLocalRows, numGlobalRows, scanLocalBlockOffsetView, scanGlobalBlockOffsetView) =
      compute_block_offsets(blocked);

  matrix->resumeFill();
  matrix->setAllToScalar(0.0);

  LO numRows = matrix->getRowMap()->getLocalNumElements();
  using matrix_execution_space =
      typename Tpetra::CrsMatrix<ST, LO, GO, NT>::local_matrix_device_type::execution_space;
  auto mat_dev = matrix->getLocalMatrixDevice();
  auto colMap_dev = matrix->getColMap()->getLocalMap();

  Kokkos::View<local_matrix_type **, Kokkos::SharedSpace> subblockMatrices;
  Kokkos::View<local_map_type **, Kokkos::SharedSpace> subblockColMaps;
  std::tie(subblockMatrices, subblockColMaps) =
      extract_view_of_local_subblock_matrices_and_maps(blocked);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, matrix_execution_space>(0, numRows),
      KOKKOS_LAMBDA(const LO localRow) {
        const auto blockRow = compute_block_row(scanLocalBlockOffsetView, localRow);

        const auto lid = localRow - scanLocalBlockOffsetView(blockRow);
        for (int blockCol = 0; blockCol < nBlockCols; ++blockCol)
        {
          const GO gidStart = scanGlobalBlockOffsetView(blockCol);
          const auto sparseRowView = subblockMatrices(blockRow, blockCol).row(lid);
          const auto map_ij = subblockColMaps(blockRow, blockCol);
          for (auto col = 0; col < sparseRowView.length; col++)
          {
            auto value = sparseRowView.value(col);
            const auto gid_ij = map_ij.getGlobalElement(sparseRowView.colidx(col));
            const auto gid = gid_ij + gidStart;
            auto colId = colMap_dev.getLocalElement(gid);
            mat_dev.sumIntoValues(localRow, &colId, 1, &value, true, false);
          }
        }
      });

  Teko::LinearOp blockedOp_lo = blocked;
  auto blockedOp_tpetra =
      Teuchos::rcp(new Teko::TpetraHelpers::TpetraOperatorWrapper(blockedOp_lo));
  auto rangeMap = blockedOp_tpetra->getRangeMap();
  auto domainMap = blockedOp_tpetra->getDomainMap();

  matrix->fillComplete(domainMap, rangeMap);
}

Teko::BlockedLinearOp extract_subblock_matrix(
    const Teko::BlockedLinearOp & blo, const std::vector<int> & rows, const std::vector<int> & cols)
{

  // allocate new operator
  auto subblock = Teko::createBlockedOp();

  const int nrows = rows.size();
  const int ncols = cols.size();

  // build new operator
  subblock->beginBlockFill(nrows, ncols);
  for (int row_id = 0U; row_id < nrows; ++row_id)
  {
    for (int col_id = 0U; col_id < ncols; ++col_id)
    {
      auto [row, col] = std::make_tuple(rows[row_id], cols[col_id]);
      auto A_row_col = blo->getBlock(row, col);

      if (A_row_col != Teuchos::null) subblock->setBlock(row_id, col_id, A_row_col);
    }
  }

  subblock->endBlockFill();

  return subblock;
}

Teuchos::RCP<const Teuchos::Comm<int>>
extract_communicator(const Teko::BlockedMultiVector & blocked)
{
  auto mv = Thyra::TpetraOperatorVectorExtraction<>::getConstTpetraMultiVector(
      Teko::getBlock(0, blocked));
  return mv->getMap()->getComm();
}

Teko::MultiVector convert_block_vector_to_vector(const Teko::BlockedMultiVector & blocked)
{
  auto tComm = extract_communicator(blocked);
  auto map = Teko::TpetraHelpers::thyraVSToTpetraMap(
      *blocked->productSpace(), Thyra::convertTpetraToThyraComm(tComm));
  auto mv = rcp(new Tpetra::MultiVector<>(map, 1));
  Teko::TpetraHelpers::blockThyraToTpetra(blocked, *mv);
  return Thyra::createMultiVector(mv);
}

Teuchos::RCP<Tpetra::Map<>> convert_block_range_to_tpetra_range(
    Teuchos::RCP<const Thyra::ProductVectorSpaceBase<Teko::ST>> range,
    Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
  return Teko::TpetraHelpers::thyraVSToTpetraMap(*range, Thyra::convertTpetraToThyraComm(comm));
}

} // namespace experimental