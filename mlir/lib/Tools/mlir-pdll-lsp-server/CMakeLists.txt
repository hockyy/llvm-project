llvm_add_library(MLIRPdllLspServerLib
  LSPServer.cpp
  PDLLServer.cpp
  Protocol.cpp
  MlirPdllLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-pdll-lsp-server

  LINK_COMPONENTS
  SupportLSP

  LINK_LIBS PUBLIC
  MLIRPDLLCodeGen
  MLIRPDLLParser
  MLIRLspServerSupportLib
  )

# Required when BUILD_SHARED_LIBS=ON so installed mlir-pdll-lsp-server can find
# the shared library (see llvm_add_library defaults).
add_mlir_library_install(MLIRPdllLspServerLib)
