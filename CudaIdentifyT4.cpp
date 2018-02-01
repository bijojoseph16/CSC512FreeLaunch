//@author Karan Jadhav
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Core/Rewriter.h"
// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PPCallbacks.h>
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include <string>
#include <iostream>
#include <iterator>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;
using namespace clang::ast_matchers;
using namespace std;


// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

std::vector<const FunctionDecl *> kernelFuncs;

clang::CUDAKernelCallExpr * hasCudaKernelCallExpr(Stmt *s) {
  Stmt * ckce;
  iterator_range<StmtIterator> s_children = s->children();
  for(StmtIterator child = s_children.begin(); child != s_children.end(); child++) {
    if(*child != NULL) {
      if(isa<CUDAKernelCallExpr>(*child)) {
	return (CUDAKernelCallExpr*) *child;
      }
      ckce = hasCudaKernelCallExpr(*child);
      if(ckce != NULL && isa<CUDAKernelCallExpr>(ckce)) {
	return (CUDAKernelCallExpr*) ckce;
      }
    }
  }
  return NULL;
}

std::string getCudaKernelCallExprName(Stmt *c) {
  std::string ckceName;
  iterator_range<StmtIterator> c_children = c->children();
  for(StmtIterator child = c_children.begin(); child != c_children.end(); child++) {
    if(*child != NULL) {
      if(isa<DeclRefExpr>(*child)) {
        return ((DeclRefExpr*) *child)->getNameInfo().getName().getAsString();
      }
      ckceName = getCudaKernelCallExprName(*child);
      if(ckceName != "") {
        return ckceName;
      }
    }
  }
  return "";
}

std::string replace(std::string &str,const std::string &strToReplace,const std::string &strToReplaceWith){
    if (str.find(strToReplace)!=std::string::npos) {
      return(str.replace(str.find(strToReplace), strToReplace.length(), strToReplaceWith));
    }
    return str;
}

std::string nodeToSourceCode(Stmt *s, SourceManager &sm) {
  return Lexer::getSourceText(CharSourceRange::getTokenRange(s->getSourceRange()), sm, LangOptions()).str();
}

class KernelFuncDefPrinter : public MatchFinder::MatchCallback {
public :
  KernelFuncDefPrinter(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    //Get parent and child funcs and calls
    const CUDAKernelCallExpr *parentCall = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>("parentCudaCall");
    const FunctionDecl *parentFunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("parentFuncDecl");
    const CUDAKernelCallExpr *childCall = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>("childCudaCall");
    const FunctionDecl *childFunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("childFuncDecl");
    const DeclStmt* globalID = Result.Nodes.getNodeAs<clang::DeclStmt>("globalID");

    //Get child blocks and threads
    CXXConstructExpr *b = (CXXConstructExpr*) childCall->getConfig()->getArg(0);
    CXXConstructExpr *t = (CXXConstructExpr*) childCall->getConfig()->getArg(1);
    std::string childBlocks = nodeToSourceCode((Stmt*) b, Rewrite.getSourceMgr());
    std::string childThreads = nodeToSourceCode((Stmt*) t, Rewrite.getSourceMgr());

    //Get global ID var from child func
    std::string global_id = ((VarDecl*)(*(globalID->decl_begin())))->getNameAsString();
    std::string global_id_type = ((VarDecl*)(*(globalID->decl_begin())))->getType().getAsString();
    
    //FL_childGridThreads
    std::string fl_cgt = "FL_childGridThreads";
    const std::string fl_cgt_init = "int " + fl_cgt + " = int(" + childBlocks + ")*" + childThreads + ";";
    //for loop
    std::string for_init = global_id_type + " " + global_id + " = 0; ";
    std::string for_end = global_id + " < " + fl_cgt + "; ";
    std::string for_inc = "++" + global_id;
    std::string for_loop = "for (" + for_init + for_end + for_inc + ") ";

    //child body to string
    Stmt* childFuncBody = childFunc->getBody();
    std::string childBodyStr = nodeToSourceCode(childFuncBody, Rewrite.getSourceMgr());
    //globalVar to string
    std::string globalIDStr = nodeToSourceCode((Stmt*) globalID, Rewrite.getSourceMgr());
    const std::string empty = "";
    //remove globalID from child Body
    replace(childBodyStr, globalIDStr, empty);
    //combine final code from fl_childGridThreads and the childBody inside the for loop
    std::string finalCode = fl_cgt_init + for_loop + childBodyStr;

    //get range of child kernel call and replace with generated code
    SourceLocation ck_s = childCall->getLocStart();
    SourceLocation ck_e = childCall->getLocEnd();
    SourceLocation ck_sc = clang::Lexer::findLocationAfterToken(ck_e, tok::semi, Rewrite.getSourceMgr(), LangOptions(), false);
    StringRef final_code_sref = StringRef(finalCode);
    Rewrite.ReplaceText(SourceRange(ck_s, ck_sc), final_code_sref);

    //remove child function declaration
    SourceLocation cfs = Rewrite.getSourceMgr().getFileLoc(childFunc->getLocStart());
    SourceLocation cfe = Rewrite.getSourceMgr().getFileLoc(childFunc->getLocEnd());
    Rewrite.RemoveText(SourceRange(cfs, cfe));
  }

private:
  Rewriter &Rewrite;
};


//OLD WORKING CODE
/*
class KernelFuncDefPrinter : public MatchFinder::MatchCallback {
public :
  KernelFuncDefPrinter(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    const FunctionDecl *kf = Result.Nodes.getNodeAs<clang::FunctionDecl>("kernelFunc");
    CUDAKernelCallExpr *ck = hasCudaKernelCallExpr(kf->getBody());
    if(ck == NULL) {
      kernelFuncs.push_back(kf);
    }
    else {
      std::string childKernelName = getCudaKernelCallExprName((Stmt*) ck);
      for(std::vector<const FunctionDecl *>::iterator f = kernelFuncs.begin(); f != kernelFuncs.end(); f++) {
	if((*f)->getNameInfo().getName().getAsString() == childKernelName) {

	  //get parameters from child kernel call
	  std::string blocks = nodeToSourceCode(ck->getConfig()->getArg(0), Rewrite.getSourceMgr());
	  std::string threads = nodeToSourceCode(ck->getConfig()->getArg(1), Rewrite.getSourceMgr());
	  //(*f)->dumpColor();
	  cout << ((VarDecl*) *((*f)->getBody()->child_begin()))->getNameAsString() << endl;;

	  //get thread/global index variable. ASSUME it is the first statement
	  
	  CXXConstructExpr *b = (CXXConstructExpr*) ck->getConfig()->getArg(0);
	  CXXConstructExpr *t = (CXXConstructExpr*) ck->getConfig()->getArg(1);
	  const SourceManager *sm = Result.SourceManager;
	  cout << "blocks : " << Lexer::getSourceText(CharSourceRange::getTokenRange(b->getSourceRange()), *sm, LangOptions()).str() << endl;
	  cout << "threads : " << Lexer::getSourceText(CharSourceRange::getTokenRange(t->getSourceRange()), *sm, LangOptions()).str() << endl;

	  //This works to remove child function!
	  Rewrite.RemoveText(SourceRange(Rewrite.getSourceMgr().getFileLoc((*f)->getLocStart()), Rewrite.getSourceMgr().getFileLoc((*f)->getLocEnd())));

	  break;
	}
      }
    }
  }

private:
  Rewriter &Rewrite;
};
*/

class MyASTConsumer: public ASTConsumer {
public:
  MyASTConsumer (Rewriter &R) : KernelFuncPrinter(R) {
    Finder.addMatcher(KernelFuncMatcher, &KernelFuncPrinter);
  }
  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Finder.matchAST(Context);
  }

private:
  KernelFuncDefPrinter KernelFuncPrinter;

  //New Matcher
  StatementMatcher KernelFuncMatcher = cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr(callee(functionDecl(hasBody(compoundStmt(hasDescendant(declStmt(has(varDecl(hasInitializer(expr())))).bind("globalID"))))).bind("childFuncDecl"))).bind("childCudaCall"))).bind("parentFuncDecl"))).bind("parentCudaCall");
  //StatementMatcher KernelFuncMatcher = cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr(callee(functionDecl(hasBody(compoundStmt(hasDescendant(declStmt().bind("globalID"))))).bind("childFuncDecl"))).bind("childCudaCall"))).bind("parentFuncDecl"))).bind("parentCudaCall");
  //StatementMatcher KernelFuncMatcher = cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr(callee(functionDecl().bind("childFuncDecl"))).bind("childCudaCall"))).bind("parentFuncDecl"))).bind("parentCudaCall");

  //Old Matcher
  //DeclarationMatcher KernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("kernelFunc");

  MatchFinder Finder;
};

class IncludeFinder : public clang::PPCallbacks {
public:
  IncludeFinder (const clang::CompilerInstance &compiler) : compiler(compiler) {
    const clang::FileID mainFile = compiler.getSourceManager().getMainFileID();
    cout << "mainFile" << endl;
    name = compiler.getSourceManager().getFileEntryForID(mainFile)->getName();
    cout << name << endl;
  }

private:
  const clang::CompilerInstance &compiler;
  std::string name;
};

class IncludeFinderAction : public PreprocessOnlyAction {
public:
  IncludeFinderAction() {}

  void ExecuteAction() override {
    IncludeFinder includeFinder(getCompilerInstance());
    cout << "includeFinder" << endl;
    getCompilerInstance().getPreprocessor().addPPCallbacks((std::unique_ptr<clang::PPCallbacks>) (&includeFinder));
    cout << "addPPCallbacks" << endl;

    clang::PreprocessOnlyAction::ExecuteAction();
    cout << "ExecuteAction" << endl;
  }
};


// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    //std::error_code error_code;
    //llvm::raw_fd_ostream outFile("/home/ubuntu/final_package/lonestargpu-2.0/apps/mst_dp_T4_modular/main.cu", error_code, llvm::sys::fs::F_None);
    //TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(outFile);
    //outFile.close();
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
  }
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
						 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};


int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
		 OptionsParser.getSourcePathList());

  //return Tool.run(newFrontendActionFactory<IncludeFinderAction>().get());
  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}

