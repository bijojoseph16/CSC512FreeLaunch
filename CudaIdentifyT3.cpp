//To run this tool on mst_dp_modular/main.cu use the following command
// bin/cuda-identify-T3 ../../Project2_freeLaunch/Examples/mst_dp_modular/main.cu -- --cuda-host-only --cuda-gpu-arch=sm_35 -w -I/usr/local/cuda/include-pthread -I ../../lonestargpu-2.0/include/ -I../../lonestargpu-2.0/cub-1.7.4/
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/AST/RecursiveASTVisitor.h"

// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "clang/Lex/Lexer.h"
#include <string>
#include <sstream>
#include <iostream>
#include <iterator>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;
using namespace clang::ast_matchers;
using namespace std;
using namespace clang::driver;


// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

map<string, string> parentVars;
vector<string> parentVar;

std::string replace(std::string &str,const std::string &strToReplace,const std::string &strToReplaceWith){
    if (str.find(strToReplace)!=std::string::npos) {
      return(str.replace(str.find(strToReplace), strToReplace.length(), strToReplaceWith));
    }
    return str;
}

class CudaCallExpressionCallback : public MatchFinder::MatchCallback {
  public:
  CudaCallExpressionCallback(Rewriter &Rewrite) : Rewrite(Rewrite) {}
  virtual void run(const MatchFinder::MatchResult &Result) {
    const CUDAKernelCallExpr *parentKernelCall = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>("parentCudaCall");
    const FunctionDecl *parentFuncDecl = Result.Nodes.getNodeAs<clang::FunctionDecl>("parentFuncDecl");
  
    //child kernel call expression
    const CUDAKernelCallExpr *childKernelCall = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>("childCudaCall");
  
    ASTContext *Context = Result.Context;
    const SourceManager *SM = Result.SourceManager;
    
    const FunctionDecl *childFuncDecl = childKernelCall->getDirectCallee();
    //Insert header for freelauch transformation
    SourceLocation startOfFile = Rewrite.getSourceMgr().getLocForStartOfFile(Rewrite.getSourceMgr().getMainFileID());
    //FullSourceLoc FullLocation = Context->getFullLoc(startOfFile);
    //llvm::outs() << "Found declaration at "<< FullLocation.getSpellingLineNumber() << ":"<< FullLocation.getSpellingColumnNumber() << "\n";
    Rewrite.InsertText(startOfFile, "#include \"freeLaunch_T3.h\"\n",true,false);
    Rewrite.InsertText(startOfFile, "#define _Bool bool \n",true,false);
  
    //In main.cu
    //Remove the child function definition
    SourceLocation startChildFuncDecl = Rewrite.getSourceMgr().getFileLoc(childFuncDecl->getLocStart());
    SourceLocation endChildFuncDecl = Rewrite.getSourceMgr().getFileLoc(childFuncDecl->getLocEnd());
    Rewrite.RemoveText(SourceRange(startChildFuncDecl, endChildFuncDecl));
    
    //In verify.cu
    //Add extra parameters to parent fucntion definition
    //last argument has getLocEnd points to the second last token
    //So we need to get the length of the parameter, to set offset to add additional parameters
    string lastParam = parentFuncDecl->parameters()[parentFuncDecl->getNumParams() - 1]->getQualifiedNameAsString();
    int offset = lastParam.length();
    SourceLocation endParamLoc = parentFuncDecl->parameters()[parentFuncDecl->getNumParams() - 1]->getLocEnd().getLocWithOffset(offset);
    string newParameters =  ", int blocks, char *FL_Args";
    Rewrite.InsertText(endParamLoc, newParameters,true,true);
        
    //In callVerify.cu
    //Initialize the FL arguments and assign it memory
    string intializeBeforeParentKernelCall = "char *FL_Arguments;\n cudaMalloc((void **)&FL_Arguments,MAX_FL_ARGSZ);\ncudaMemset(FL_Arguments,0,MAX_FL_ARGSZ);\n";
    Rewrite.InsertText(parentKernelCall->getLocStart(), intializeBeforeParentKernelCall,true,false);
    //Get the parent kernel parameters
    SourceRange range = parentKernelCall->getConfig()->getArg(0)->getSourceRange();
    llvm::StringRef numBlocks = Lexer::getSourceText(CharSourceRange::getTokenRange(range), *SM, LangOptions());
    //Now insert these new paramters into parent function call
    string newParametersForParentCall =  ", " + numBlocks.str() + ", FL_Arguments";
    Rewrite.InsertText(parentKernelCall->getLocEnd(), newParametersForParentCall,true,true);
    //Insert statement to free memory allocated to arguments
    SourceLocation locEndOfParentCall = clang::Lexer::findLocationAfterToken(parentKernelCall->getLocEnd(), tok::semi, *SM, LangOptions(), false);
    string deallocateAfterParentKernelCall = "\ncudaFree(FL_Arguments);";
    Rewrite.InsertText(locEndOfParentCall, deallocateAfterParentKernelCall,true,false);

    //In verify.cu
    //Add the PRELOOP Statement
    //SourceLocation startLocOfParentFunc = clang::Lexer::findLocationAfterToken(parentFuncDecl->getLocEnd(), tok::l_brace, *SM, LangOptions(), false);
    Rewrite.InsertTextAfterToken(parentFuncDecl->getBody()->getLocStart(), "\nFL_T3_Preloop;\n");

    //Remove the child kernel and and add code to record kernel info
    SourceRange childGridrange = childKernelCall->getConfig()->getArg(0)->getSourceRange();
    llvm::StringRef childGridSize = Lexer::getSourceText(CharSourceRange::getTokenRange(childGridrange), *SM, LangOptions());

    SourceRange childBlockrange = childKernelCall->getConfig()->getArg(1)->getSourceRange();
    llvm::StringRef childBlockSize = Lexer::getSourceText(CharSourceRange::getTokenRange(childBlockrange ), *SM, LangOptions());

    string FL_bllc = "int FL_bllc = atomicAdd(&FL_blc,1);\n";
    string FL_childBlockSize = "FL_childBlockSize = " +childBlockSize.str()+";\n";
    string FL_childKernelArgSz = "FL_childKernelArgSz = sizeof(int)";
    map <string, string> :: iterator itr;
    int argC = 0;
    for (itr = parentVars.begin(); itr != parentVars.end(); ++itr)
    {
        FL_childKernelArgSz += "+sizeof("+itr->second+")";
    }
    FL_childKernelArgSz += ";";
    string _tmp_p = "\nchar * _tmp_p = (char *) (FL_pArgs+FL_bllc*FL_childKernelArgSz);\n";
    string _tmp_childGridSize = "int _tmp_childGridSize = "+ childGridSize.str() + ";";
    
    //Copy all the childKernelVariables
    string _tmp_pmcpy = "\nmemcpy((void*)_tmp_p, (void*) &_tmp_childGridSize, sizeof(int));\n_tmp_p+=sizeof(int);\n";
    /*
    for (itr = parentVars.begin(); itr != parentVars.end(); ++itr)
    {
      _tmp_pmcpy += "memcpy((void*)_tmp_p, (void*) &"+itr->first+", sizeof("+itr->second+"));\n_tmp_p+=sizeof("+itr->second+");\n";
    }
    */
    for(unsigned i=0; i<parentVar.size(); i++) {
      string delimiter = ";";
      size_t pos = 0;
      pos = parentVar.at(i).find(delimiter);
      string dataType =  parentVar.at(i).substr(0, pos);
      string varName = parentVar.at(i).substr(pos + 1, parentVar.at(i).length() - pos);
      _tmp_pmcpy += "memcpy((void*)_tmp_p, (void*) &"+varName+", sizeof("+dataType+"));\n_tmp_p+=sizeof("+dataType+");\n";
    }

    string 	FL_check = "\nFL_check = 0;\ngoto P;\nC:	__threadfence();\n";

    string recordKernelInfo = FL_bllc + FL_childBlockSize + FL_childKernelArgSz + _tmp_p + _tmp_childGridSize +_tmp_pmcpy + FL_check;

    //Get Location of child kernel call and remove it
    SourceLocation locEndOfChildCall = clang::Lexer::findLocationAfterToken(childKernelCall->getLocEnd(), tok::semi, *SM, LangOptions(), false);
    Rewrite.RemoveText(SourceRange(childKernelCall->getLocStart(),locEndOfChildCall));
    
    //Now insert these parameters inplace of child kernel call
    Rewrite.InsertText(childKernelCall->getLocStart(), recordKernelInfo, true,true);
    //Add the POSTLOOP Statement
    Rewrite.InsertText(parentFuncDecl->getBody()->getLocEnd(), "FL_T3_Postloop;\n", true,false);
	
    //Add code to retrieve child kernel modified arguments
    string initialize = "char * _tmp_p = (char*)(FL_pArgs+ckernelSeqNum*FL_childKernelArgSz);\nint FL_childGridSize;\nmemcpy((void*)&FL_childGridSize, (void*)_tmp_p, sizeof(int));\n_tmp_p+=sizeof(int);\n";
  
    /*
    for (itr = parentVars.begin(); itr != parentVars.end(); ++itr) {
      initialize += itr->second +" "+ itr->first + ";\n";
      initialize += "memcpy((void*)&"+itr->first+",(void*)_tmp_p,"+"sizeof("+itr->second+"));\n";
      initialize += "_tmp_p+=sizeof("+itr->second+");\n";
    }
    */
    for(unsigned i=0; i<parentVar.size(); i++) {
      string delimiter = ";";
      size_t pos = 0;
      pos = parentVar.at(i).find(delimiter);
      string dataType =  parentVar.at(i).substr(0, pos);
      string varName = parentVar.at(i).substr(pos + 1, parentVar.at(i).length() - pos);
      initialize += dataType +" "+ varName + ";\n";
      initialize += "memcpy((void*)&"+varName+",(void*)_tmp_p,"+"sizeof("+dataType+"));\n";
      initialize += "_tmp_p+=sizeof("+dataType+");\n";
    }

    Rewrite.InsertText(parentFuncDecl->getBody()->getLocEnd(), initialize, true,false); 

    SourceRange childFuncrange = childFuncDecl->getBody()->getSourceRange();
    llvm::StringRef childBody = Lexer::getSourceText(CharSourceRange::getTokenRange(childFuncrange), *SM, LangOptions());
    string childBodyStr = childBody.str();
    replace(childBodyStr,"threadIdx.x + blockIdx.x*blockDim.x;","kkk *FL_childBlockSize +threadIdx.x%FL_childBlockSize;");
    replace(childBodyStr,"return;","continue;");
    //Add the contents of the child kernel function definition to where it is called
    string loopTohandleChildTask = "\nfor(int kkk=0;kkk< FL_childGridSize;kkk++)"+childBodyStr.substr(0, childBodyStr.length() - 1)+"\n";
    Rewrite.InsertText(parentFuncDecl->getLocEnd(), loopTohandleChildTask, true, false);
    Rewrite.InsertText(parentFuncDecl->getLocEnd(),"FL_postChildLog;\n", true, true);
  } 
  private:
  Rewriter &Rewrite;
};

class ParentReturnStatementCallback : public MatchFinder::MatchCallback {
  public:
  ParentReturnStatementCallback(Rewriter &Rewrite) : Rewrite(Rewrite) {}
  virtual void run(const MatchFinder::MatchResult &Result) {
    const ReturnStmt *returnStmtParent = Result.Nodes.getNodeAs<clang::ReturnStmt>("parentReturnStmt");
    
    //Changes to verify.cu
    //Replace return statements with goto
    SourceLocation parentReturnLocation = returnStmtParent->getLocStart();
    Rewrite.RemoveText(returnStmtParent->getSourceRange());
    Rewrite.InsertText(parentReturnLocation,"goto P",true,true);

  }
  private:
  Rewriter &Rewrite;
};

class ChildReturnStatementCallback : public MatchFinder::MatchCallback {
  public:
  ChildReturnStatementCallback(Rewriter &Rewrite) : Rewrite(Rewrite) {}
  virtual void run(const MatchFinder::MatchResult &Result) {
    const ReturnStmt *returnStmtChild = Result.Nodes.getNodeAs<clang::ReturnStmt>("childReturnStmt");

    //Replace the return statements of child with continue;
    SourceLocation childReturnLocation = returnStmtChild->getLocStart();
    Rewrite.RemoveText(returnStmtChild->getSourceRange());
    Rewrite.InsertText(childReturnLocation,"continue",true,true);
  }
  private:
  Rewriter &Rewrite;
};
class ParentVarDeclCallback : public MatchFinder::MatchCallback {
  public:
  ParentVarDeclCallback(Rewriter &Rewrite) : Rewrite(Rewrite) {}
  virtual void run(const MatchFinder::MatchResult &Result) {
    const VarDecl *parentVarDecl = Result.Nodes.getNodeAs<clang::VarDecl>("parentVarDecl");
    //Changes to verify.cu  
    //Get the variable decl
    parentVars.insert(pair<string, string>(parentVarDecl->getQualifiedNameAsString(),parentVarDecl->getType().getAsString()));
    parentVar.push_back(parentVarDecl->getType().getAsString() +";"+parentVarDecl->getQualifiedNameAsString());
  }
  private:
  Rewriter &Rewrite;
};

class MyASTConsumer: public ASTConsumer {
public:
  MyASTConsumer (Rewriter &R) : ParentVarDecl(R), ChildReturnStatement(R), CudaCallExpression(R), ParentReturnStatement(R) {
    //Finder.addMatcher(KernelFuncMatcher, &KernelFuncPrinter);
    Finder.addMatcher(ParentVarDeclMatcher, &ParentVarDecl);
    Finder.addMatcher(ChildReturnStmtMatcher, &ChildReturnStatement);
    Finder.addMatcher(CudaKernelCallMatcher, &CudaCallExpression);
    Finder.addMatcher(ParentReturnStmtMatcher, &ParentReturnStatement);
    
  }
  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Finder.matchAST(Context);
  }
private:
  //KernelFuncDefPrinter KernelFuncPrinter;
  //DeclarationMatcher KernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("kernelFunc");
  StatementMatcher ParentVarDeclMatcher = cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr().bind("childCudaCall")),hasBody(forEachDescendant(varDecl().bind("parentVarDecl")))).bind("parentFuncDecl"))).bind("parentCudaCall");   
  ParentVarDeclCallback ParentVarDecl;
  StatementMatcher ChildReturnStmtMatcher = cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr(callee(functionDecl(forEachDescendant(returnStmt().bind("childReturnStmt"))))).bind("childCudaCall"))).bind("parentFuncDecl"))).bind("parentCudaCall");   
  ChildReturnStatementCallback ChildReturnStatement;
  StatementMatcher CudaKernelCallMatcher = cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr().bind("childCudaCall"))).bind("parentFuncDecl"))).bind("parentCudaCall");   
  CudaCallExpressionCallback CudaCallExpression;
  StatementMatcher ParentReturnStmtMatcher = cudaKernelCallExpr(callee(functionDecl(hasDescendant(cudaKernelCallExpr().bind("childCudaCall")),forEachDescendant(returnStmt().bind("parentReturnStmt"))).bind("parentFuncDecl"))).bind("parentCudaCall");   
  ParentReturnStatementCallback ParentReturnStatement;

  MatchFinder Finder;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    std::error_code error_code;
    llvm::raw_fd_ostream outFile("/home/ubuntu/Project2_freeLaunch/Examples/mst_dp_modular_1/mainT3.cu", error_code, llvm::sys::fs::F_None);
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(outFile);
    outFile.close();
    //const RewriteBuffer *RewriteBuf = TheRewriter.getRewriteBufferFor(TheRewriter.getSourceMgr().getMainFileID());
   
  }
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
						 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(),CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter);
  }

  private:
  Rewriter TheRewriter;
};


int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
		 OptionsParser.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}

