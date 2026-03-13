/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#pragma once

#include "Teuchos_RCP.hpp"
#include "Teko_BlockPreconditionerFactory.hpp"
#include "Teko_Utilities.hpp"
#include "Xpetra_MultiVector_decl.hpp"
#include "MueLu_CreateTpetraPreconditioner.hpp"
#include "MueLu_MultiPhys_decl.hpp"
#include "Teko_JacobiPreconditionerFactory.hpp"
#include "Teko_GaussSeidelPreconditionerFactory.hpp"

namespace multiphys
{

class BlockAMGPreconditionerFactory : public Teko::BlockPreconditionerFactory
{
public:
  using ST = Teko::ST;
  using LO = Teko::LO;
  using GO = Teko::GO;
  using NT = Teko::NT;
  using XpetraMultiVector = Xpetra::MultiVector<ST, LO, GO, NT>;
  using XpetraMatrix = Xpetra::Matrix<ST, LO, GO, NT>;
  using XpetraBlockedMatrix = Xpetra::BlockedCrsMatrix<ST, LO, GO, NT>;
  using XpetraOperator = Xpetra::Operator<ST, LO, GO, NT>;
  using Matrix = Tpetra::CrsMatrix<>;
  BlockAMGPreconditionerFactory();
  Teko::LinearOp buildPreconditionerOperator(
      Teko::BlockedLinearOp & blo, Teko::BlockPreconditionerState & state) const final;

protected:
  void initializeFromParameterList(const Teuchos::ParameterList & pl) override;

private:
  Teuchos::RCP<Teuchos::ParameterList> amgSettings;
  std::vector<Teuchos::RCP<XpetraMultiVector>> coordinates;
  std::vector<Teuchos::RCP<XpetraMultiVector>> material;
  std::vector<Teuchos::RCP<XpetraMultiVector>> nullspace;
  bool useBlockSmoothers = true;
  bool useMaxLevels = false;
  mutable Teuchos::RCP<MueLu::MultiPhys<ST, LO, GO, NT>> multiphysPreconditioner;
  mutable Teuchos::RCP<Matrix> A_tpetra;
  mutable Teuchos::RCP<XpetraBlockedMatrix> A_blocked_xpetra;
};

} // namespace experimental
