set(LLVM_LINK_COMPONENTS
  Demangle
  Support
  TableGen
  SupportLSP
)

llvm_add_library(TableGenLspServerLib
  LSPServer.cpp
  TableGenServer.cpp
  TableGenLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/tblgen-lsp-server

  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_LIBS PUBLIC
  MLIRLspServerSupportLib
  MLIRSupport
  )

# Required when BUILD_SHARED_LIBS=ON so installed tblgen-lsp-server can find the
# shared library (see llvm_add_library defaults).
add_mlir_library_install(TableGenLspServerLib)
