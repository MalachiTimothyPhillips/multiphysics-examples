/*--------------------------------------------------------------------*/
/*    Copyright 2002 - 2008, 2010, 2011 National Technology &         */
/*    Engineering Solutions of Sandia, LLC (NTESS). Under the terms   */
/*    of Contract DE-NA0003525 with NTESS, there is a                 */
/*    non-exclusive license for use of this work by or on behalf      */
/*    of the U.S. Government.  Export of this program may require     */
/*    a license from the United States Government.                    */
/*--------------------------------------------------------------------*/

#include "teko_amg.hpp"
#include "utilities.hpp"
#include "Teko_ImplicitLinearOp.hpp"
#include "Teko_BlockImplicitLinearOp.hpp"
#include <Tpetra_Operator.hpp>
#include "MueLu_TpetraOperator_decl.hpp"
#include "MueLu_CreateTpetraPreconditioner.hpp"
#include "MueLu_MultiPhys_decl.hpp"
#include "Xpetra_BlockedCrsMatrix_decl.hpp"
#include "MueLu_SmootherBase.hpp"
#include "Xpetra_MultiVectorFactory_decl.hpp"
#include "Xpetra_TpetraCrsMatrix_decl.hpp"
#include "MueLu_Utilities_decl.hpp"
#include "Teko_BlockDiagonalInverseOp.hpp"
#include "Teuchos_StackedTimer.hpp"

namespace multiphys
{

using Teko::GO;
using Teko::LO;
using Teko::NT;
using Teko::ST;
using XpetraMultiVector = Xpetra::MultiVector<ST, LO, GO, NT>;
using XpetraMatrix = Xpetra::Matrix<ST, LO, GO, NT>;
using XpetraMap = Xpetra::Map<LO, GO, NT>;
using XpetraOperator = Xpetra::Operator<ST, LO, GO, NT>;
using XpetraTpetraMV = Xpetra::TpetraMultiVector<ST, LO, GO, NT>;
using Hierarchy = MueLu::Hierarchy<ST, LO, GO, NT>;
using Teuchos::RCP;

class WrapMueLuAsBlockOperator : public Teko::BlockImplicitLinearOp
{
public:
  WrapMueLuAsBlockOperator(Teko::BlockedLinearOp & A,
      Teuchos::RCP<MueLu::MultiPhys<ST, LO, GO, NT>> & Minv,
      bool useBlockSmoothers)
      : A_(A),
        Minv_(Minv),
        productRange_(A->productRange()),
        productDomain_(A->productDomain()),
        useBlockSmoothers_(useBlockSmoothers)
  {
  }

  [[nodiscard]] Teko::VectorSpace range() const override { return productRange_; }
  [[nodiscard]] Teko::VectorSpace domain() const override { return productDomain_; }

  void implicitApply(const Teko::BlockedMultiVector & r,
      Teko::BlockedMultiVector & z,
      const double alpha = 1.0,
      const double beta = 0.0) const override
  {
    if (useBlockSmoothers_)
    {
      auto comm = extract_communicator(A_);
      Teuchos::RCP<const Thyra::MultiVectorBase<ST>> r_cast =
          Teuchos::rcp_dynamic_cast<const Thyra::MultiVectorBase<ST>>(r);
      Teuchos::RCP<Thyra::MultiVectorBase<ST>> z_cast =
          Teuchos::rcp_dynamic_cast<Thyra::MultiVectorBase<ST>>(z);
      auto r_xpetra = Xpetra::ThyraUtils<ST, LO, GO, NT>::toXpetra(r_cast, comm);
      auto z_xpetra = Xpetra::ThyraUtils<ST, LO, GO, NT>::toXpetra(z_cast, comm);
      Minv_->apply(*r_xpetra, *z_xpetra, Teuchos::NO_TRANS, alpha, beta);
      Xpetra::ThyraUtils<ST, LO, GO, NT>::updateThyra(z_xpetra, Teuchos::null, z);
    }
    else
    {
      auto tm = Teuchos::make_rcp<Teuchos::TimeMonitor>(
          *Teuchos::TimeMonitor::getNewTimer("WrapMueLuAsBlockOperator::implicitApply"));

      {
        auto blockVecToVecTM = Teuchos::make_rcp<Teuchos::TimeMonitor>(
            *Teuchos::TimeMonitor::getNewTimer("Convert Input Block Vectors to Vector"));
        if (!flattened_r_)
        {
          flattened_r_ = convert_block_vector_to_vector(r);
          flattened_z_ = convert_block_vector_to_vector(z);
        }
        else
        {
          auto z_mv = Thyra::TpetraOperatorVectorExtraction<>::getTpetraMultiVector(flattened_z_);
          auto r_mv = Thyra::TpetraOperatorVectorExtraction<>::getTpetraMultiVector(flattened_r_);
          Teko::TpetraHelpers::blockThyraToTpetra(z, *z_mv);
          Teko::TpetraHelpers::blockThyraToTpetra(r, *r_mv);
        }
      }

      auto z_mv = Thyra::TpetraOperatorVectorExtraction<>::getTpetraMultiVector(flattened_z_);
      auto r_mv = Thyra::TpetraOperatorVectorExtraction<>::getTpetraMultiVector(flattened_r_);
      auto z_xpetra = Xpetra::toXpetra(z_mv);
      auto r_xpetra = Xpetra::toXpetra(r_mv);
      Minv_->apply(*r_xpetra, *z_xpetra, Teuchos::NO_TRANS, alpha, beta);

      {
        auto blockVecToVecTM = Teuchos::make_rcp<Teuchos::TimeMonitor>(
            *Teuchos::TimeMonitor::getNewTimer("Convert Output Vector to Block Vector"));
        Teko::TpetraHelpers::blockTpetraToThyra(*z_mv, z.ptr());
      }
    }
  }

  void implicitApply(const Thyra::EOpTransp M_trans,
      const Teko::BlockedMultiVector & r,
      Teko::BlockedMultiVector & z,
      const double alpha = 1.0,
      const double beta = 0.0) const override
  {
    if (M_trans == Thyra::NOTRANS)
    {
      this->implicitApply(r, z, alpha, beta);
    }
    else
    {
      std::ostringstream errMsg;
      errMsg << "WrapMueLuAsBlockOperator::implicitApply not implemented for M_trans != "
                "Thyra::NOTRANS\n";
    }
  }

private:
  Teko::BlockedLinearOp A_;
  Teuchos::RCP<MueLu::MultiPhys<ST, LO, GO, NT>> Minv_;
  Teko::VectorSpace productRange_;
  Teko::VectorSpace productDomain_;
  bool useBlockSmoothers_;
  mutable Teko::MultiVector flattened_r_;
  mutable Teko::MultiVector flattened_z_;
};

BlockAMGPreconditionerFactory::BlockAMGPreconditionerFactory() : Teko::BlockPreconditionerFactory()
{
}

Teko::LinearOp BlockAMGPreconditionerFactory::buildPreconditionerOperator(
    Teko::BlockedLinearOp & blo, Teko::BlockPreconditionerState & state) const
{
  const int nBlk = Teko::blockRowCount(blo);

  if (!useBlockSmoothers)
  {
    if (A_tpetra)
    {
      util::update_block_matrix_to_crs_matrix(blo, A_tpetra);
    }
    else
    {
      A_tpetra = util::convert_block_matrix_to_crs_matrix(blo);
    }
  }
  else
  {
    if (A_blocked_xpetra)
    {
      for (int row = 0; row < nBlk; ++row)
      {
        for (int col = 0; col < nBlk; ++col)
        {
          A_blocked_xpetra->setMatrix(row,
              col,
              Xpetra::toXpetra(Teuchos::rcp_const_cast<Tpetra::CrsMatrix<ST, LO, GO, NT>>(
                  get_crs_matrix(row, col, blo))));
        }
      }
    }
    else
    {
      A_blocked_xpetra =
          Teuchos::make_rcp<Xpetra::BlockedCrsMatrix<ST, LO, GO, NT>>(blo, Teuchos::null);
      A_blocked_xpetra->fillComplete();
    }
  }

  Teuchos::ArrayRCP<Teuchos::RCP<XpetraMatrix>> arrayOfAuxMatrices(nBlk);
  Teuchos::ArrayRCP<Teuchos::RCP<XpetraMultiVector>> arrayOfCoords(nBlk);
  Teuchos::ArrayRCP<Teuchos::RCP<XpetraMultiVector>> arrayOfMaterials(nBlk);
  Teuchos::ArrayRCP<Teuchos::RCP<XpetraMultiVector>> arrayOfNullspaces(nBlk);

  for (int blk = 0; blk < nBlk; ++blk)
  {
    ST scalar{0.0};
    bool transpose{false};
    auto subblockMat = Teuchos::rcp_const_cast<Tpetra::CrsMatrix<ST, LO, GO, NT>>(
        Teko::TpetraHelpers::getTpetraCrsMatrix(
            Teko::getBlock(blk, blk, blo), &scalar, &transpose));
    arrayOfAuxMatrices[blk] = Xpetra::toXpetra(subblockMat);
    if (coordinates[blk])
    {
      arrayOfCoords[blk] = coordinates[blk];
    }

    if (material[blk])
    {
      arrayOfMaterials[blk] = material[blk];
    }

    if (nullspace[blk])
    {
      arrayOfNullspaces[blk] = nullspace[blk];
    }
  }

  Teuchos::RCP<XpetraMatrix> A_blo = A_blocked_xpetra;
  Teuchos::RCP<XpetraMatrix> A_xpetra = useBlockSmoothers ? A_blo : Xpetra::toXpetra(A_tpetra);

  if (multiphysPreconditioner)
  {
    multiphysPreconditioner->resetMatrix(A_xpetra, true);
  }
  else
  {
    multiphysPreconditioner = Teuchos::make_rcp<MueLu::MultiPhys<ST, LO, GO, NT>>(A_xpetra,
        arrayOfAuxMatrices,
        arrayOfNullspaces,
        arrayOfCoords,
        nBlk,
        *amgSettings,
        true, // compute preconditioner now
        arrayOfMaterials,
        true // always omit subblock solver setup, since Teko-based preconditioners will ultimately
             // be used here
    );
  }

  return Teuchos::rcp(
      new WrapMueLuAsBlockOperator(blo, multiphysPreconditioner, useBlockSmoothers));
}

namespace
{
bool detect_block_smoother(const Teuchos::ParameterList & pl)
{
  for (auto && param : pl)
  {
    bool smootherTypeSpecification = param.key.find("smoother:") != std::string::npos;
    smootherTypeSpecification &= param.key.find("type") != std::string::npos;
    if (!smootherTypeSpecification) continue;

    if (pl.get<std::string>(param.key).find("teko") != std::string::npos) return true;
  }
  return false;
}
} // namespace

void BlockAMGPreconditionerFactory::initializeFromParameterList(const Teuchos::ParameterList & pl)
{
  auto settings = pl.sublist("AMG Settings");
  amgSettings = Teuchos::make_rcp<Teuchos::ParameterList>(settings);

  int numBlocks = 0;
  for (auto && param : *amgSettings)
  {
    if (!amgSettings->isSublist(param.key)) continue;
    if (param.key.find("subblockList") != std::string::npos) numBlocks++;
  }

  if (amgSettings->isParameter("combine: useMaxLevels"))
  {
    useMaxLevels = amgSettings->get<bool>("combine: useMaxLevels");
  }

  coordinates.resize(numBlocks);
  material.resize(numBlocks);
  nullspace.resize(numBlocks);

  if (pl.isSublist("user data"))
  {
    const auto & user_data = pl.sublist("user data");
    for (int block = 0; block < numBlocks; ++block)
    {
      const auto coordsName = "Coordinates" + std::to_string(block);
      if (user_data.isParameter(coordsName))
      {
        coordinates[block] = user_data.get<Teuchos::RCP<XpetraMultiVector>>(coordsName);
      }

      const auto materialName = "Material" + std::to_string(block);
      if (user_data.isParameter(materialName))
      {
        material[block] = user_data.get<Teuchos::RCP<XpetraMultiVector>>(materialName);
      }

      const auto nullspaceName = "Nullspace" + std::to_string(block);
      if (user_data.isParameter(nullspaceName))
      {
        nullspace[block] = user_data.get<Teuchos::RCP<XpetraMultiVector>>(nullspaceName);
      }
    }
  }

  useBlockSmoothers = detect_block_smoother(*amgSettings);

  if (useBlockSmoothers)
  {
    const bool useSubcommunicators = false;
    amgSettings->set("repartition: use subcommunicators", useSubcommunicators);
    for (int subblock = 0; subblock < numBlocks; ++subblock)
    {
      auto & subblockParams = amgSettings->sublist("subblockList" + std::to_string(subblock), true);
      subblockParams.set("repartition: use subcommunicators", useSubcommunicators);
    }
  }
}

} // namespace experimental
