/** Minimum spanning tree -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @Description
 * Computes minimum spanning tree of a graph using Boruvka's algorithm.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

#include "lonestargpu.h"
#include "gbar.cuh"
#include "cuda_launch_config.hpp"
#include "devel.h"
__device__ unsigned user_getOutDegree(unsigned int* noutgoing,unsigned nnodes,unsigned src) {
        if (src < nnodes) {
                return noutgoing[src];
        }
        unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
        printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
        return 0;
}
__device__ unsigned user_getInDegree(unsigned int* nincoming,unsigned nnodes,unsigned dst) {
        if (dst < nnodes) {
                return nincoming[dst];
        }
        unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
        printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, dst, nnodes);
        return 0;
}
__device__ unsigned user_getFirstEdge(unsigned *noutgoing,unsigned int* srcsrc,unsigned int* psrc,unsigned int nnodes,unsigned src);
__device__ unsigned user_getDestination(unsigned *noutgoing,unsigned *edgessrcdst,unsigned int* srcsrc,unsigned *psrc,unsigned int nnodes,unsigned nedges,unsigned src, unsigned nthedge) {
        unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
        if (src < nnodes && nthedge < user_getOutDegree(noutgoing,nnodes,src)) {
                unsigned edge = user_getFirstEdge(noutgoing,srcsrc,psrc,nnodes,src) + nthedge;
                if (edge && edge < nedges + 1) {
                        return edgessrcdst[edge];
                }
                ////printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, edge, nedges + 1);
                return nnodes;
        }
      /*  if (src < nnodes) {
                printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nthedge, getOutDegree(src));
        } else {
                printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
        }*/
        return nnodes;
}
__device__ foru user_getWeight(unsigned *noutgoing,foru* edgessrcwt,unsigned *srcsrc,unsigned *psrc,unsigned int nnodes,unsigned nedges,unsigned src, unsigned nthedge) {
        unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
        if (src < nnodes && nthedge < user_getOutDegree(noutgoing,nnodes,src)) {
                unsigned edge = user_getFirstEdge(noutgoing,srcsrc,psrc,nnodes,src) + nthedge;
                if (edge && edge < nedges + 1) {
                        return edgessrcwt[edge];
                }
                ////printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, edge, nedges + 1);
                return MYINFINITY;
        }
       /* if (src < nnodes) {
                printf("Error: %s(%d): thread %d, node %d: edge %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nthedge, getOutDegree(src));
        } else {
                printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
        }*/
        return MYINFINITY;
}

__device__ unsigned user_getFirstEdge(unsigned *noutgoing,unsigned int* srcsrc,unsigned int* psrc,unsigned int nnodes,unsigned src) {
        unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
        if (src < nnodes) {
                unsigned srcnout = user_getOutDegree(noutgoing,nnodes,src);
		if (srcnout > 0 && srcsrc[src] < nnodes) {
                        return psrc[srcsrc[src]];
                }
                //printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, 0, srcnout);
                return 0;
        }
        printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
        return 0;
}
__device__ unsigned user_getMinEdge(unsigned *noutgoing,foru* edgessrcwt,unsigned *srcsrc,unsigned *psrc,unsigned nnodes,unsigned nedges,unsigned src) {
        unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
        if (src < nnodes) {
                unsigned srcnout = user_getOutDegree(noutgoing,nnodes,src);
                if (srcnout > 0) {
                        unsigned minedge = 0;
                        foru    minwt   = user_getWeight(noutgoing,edgessrcwt,srcsrc,psrc,nnodes,nedges,src, 0);
                        for (unsigned ii = 1; ii < srcnout; ++ii) {
                                foru wt = user_getWeight(noutgoing,edgessrcwt,srcsrc,psrc,nnodes,nedges,src, ii);
                                if (wt < minwt) {
                                        minedge = ii;
                                        minwt = wt;
                                }
                        }
                        return minedge;
                }
                printf("Error: %s(%d): thread %d, edge %d out of bounds %d.\n", __FILE__, __LINE__, id, 0, srcnout);
                return 0;
        }
        printf("Error: %s(%d): thread %d, node %d out of bounds %d.\n", __FILE__, __LINE__, id, src, nnodes);
        return 0;
}

__global__ void dinit(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		eleminwts[id] = MYINFINITY;
		minwtcomponent[id] = MYINFINITY;	
		goaheadnodeofcomponent[id] = graph.nnodes;
		phores[id] = 0;
		partners[id] = id;
		processinnextiteration[id] = false;
	}
}
__device__ bool isBoss(unsigned *ele2comp,unsigned element) {
  return atomicCAS(&ele2comp[element],element,element) == element;
}

__device__ unsigned user_find(unsigned int* ele2comp,unsigned lelement, bool compresspath= true) {
        // do we need to worry about concurrency in this function?
        // for other finds, no synchronization necessary as the data-structure is a tree.
        // for other unifys, synchornization is not required considering that unify is going to affect only bosses, while find is going to affect only non-bosses.
        unsigned element = lelement;
        while (isBoss(ele2comp,element) == false) {
          element = ele2comp[element];
        }
        if (compresspath) ele2comp[lelement] = element; // path compression.
        return element;
}

__global__ void dfindelemin(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
		// if I have a cross-component edge,
		// 	find my minimum wt cross-component edge,
		//	inform my boss about this edge e (atomicMin).
		unsigned src = id;
		unsigned srcboss = cs.find(src);
		unsigned dstboss = graph.nnodes;
		foru minwt = MYINFINITY;
		unsigned degree = graph.getOutDegree(src);
		for (unsigned ii = 0; ii < degree; ++ii) {
			foru wt = graph.getWeight(src, ii);
			if (wt < minwt) {
				unsigned dst = graph.getDestination(src, ii);
				unsigned tempdstboss = cs.find(dst);
				if (srcboss != tempdstboss) {	// cross-component edge.
					minwt = wt;
					dstboss = tempdstboss;
				}
			}
		}
		dprintf("\tminwt[%d] = %d\n", id, minwt);
		eleminwts[id] = minwt;
		partners[id] = dstboss;

		if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {
			// inform boss.
			foru oldminwt = atomicMin(&minwtcomponent[srcboss], minwt);
			// if (oldminwt > minwt && minwtcomponent[srcboss] == minwt)
			//   {			    
			
			// 	goaheadnodeofcomponent[srcboss],id);	// threads with same wt edge will race.
			// 	dprintf("\tpartner[%d(%d)] = %d init, eleminwts[id]=%d\n", id, srcboss, dstboss, eleminwts[id]);
			//   }
		}
	}
}

__global__ void dfindelemin2(unsigned *mstwt, unsigned nnodes,unsigned nedges,unsigned *noutgoing,unsigned *nincoming,unsigned *edgessrcdst,foru *edgessrcwt, unsigned *srcsrc,unsigned *psrc,
 unsigned *ele2comp, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (id < nnodes) {
		unsigned src = id;
		unsigned srcboss = user_find(ele2comp,src);//cs.find(src);

		if(eleminwts[id] == minwtcomponent[srcboss] && srcboss != partners[id] && partners[id] != nnodes)
		  {	//printf("%d %d\n",id,threadIdx.x);
		    unsigned degree = user_getOutDegree(noutgoing,nnodes,src);
		    for (unsigned ii = 0; ii < degree; ++ii) {
		      foru wt = user_getWeight(noutgoing,edgessrcwt,srcsrc,psrc,nnodes,nedges,src, ii);
		      if (wt == eleminwts[id]) {
			unsigned dst = user_getDestination(noutgoing,edgessrcdst,srcsrc,psrc,nnodes,nedges,src, ii);
			unsigned tempdstboss = user_find(ele2comp,dst);//cs.find(dst);
			if (tempdstboss == partners[id]) {	// cross-component edge.
			  //atomicMin(&goaheadnodeofcomponent[srcboss], id);
			  
			  if(atomicCAS(&goaheadnodeofcomponent[srcboss], nnodes, id) == nnodes)
			    {
			      //printf("%d: adding %d\n", id, eleminwts[id]);
			      //atomicAdd(wt2, eleminwts[id]);
			    }
			}
		      }
		    }
		  }
	}
}

//#include "child.cu"
//Remove the include statement for chid.cu and added contents of file
__global__ void child(unsigned *noutgoing,unsigned *ele2comp,unsigned *partners,unsigned id,foru *edgessrcwt,unsigned *edgessrcdst,unsigned *srcsrc,unsigned *psrc,bool *processinnextiteration,unsigned nnodes,unsigned nedges,unsigned minwt_node,foru minwt,bool minwt_found,unsigned degree)
{
  unsigned ii = threadIdx.x + blockIdx.x*blockDim.x;
  if(ii<degree){
    foru wt = user_getWeight(noutgoing,edgessrcwt,srcsrc,psrc,nnodes,nedges,minwt_node, ii);
    if (wt == minwt) {
      minwt_found = true;
      unsigned dst = user_getDestination(noutgoing,edgessrcdst,srcsrc,psrc,nnodes,nedges,minwt_node, ii);
      unsigned tempdstboss = user_find(ele2comp,dst);
      if(tempdstboss == partners[minwt_node] && tempdstboss != id)
	{
	  processinnextiteration[minwt_node] = true;
	  return;
	}
    }

  } 
}

//#include "verify.cu"
//Remove the include statement and add the function body of verify.cu
__global__ void verify_min_elem(unsigned *mstwt, unsigned nnodes,unsigned nedges,unsigned *noutgoing,unsigned *nincoming,unsigned *edgessrcdst,foru *edgessrcwt, unsigned *srcsrc,unsigned *psrc,unsigned *ele2comp, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid  ) {

  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (inpid < nnodes) id = inpid;
  if (id < nnodes) {
    if(isBoss(ele2comp,id))
      {
	if(goaheadnodeofcomponent[id] == nnodes)
	  {
	    return;
	  }

	unsigned minwt_node = goaheadnodeofcomponent[id];
	unsigned degree = user_getOutDegree(noutgoing,nnodes,minwt_node);
	foru minwt = minwtcomponent[id];
	if(minwt == MYINFINITY)
	  return;
	bool minwt_found = false;
	child<<<(degree+31)/32,32>>>(noutgoing,ele2comp,partners,id,edgessrcwt,edgessrcdst,srcsrc,psrc,processinnextiteration,nnodes,nedges,minwt_node,minwt,minwt_found,degree); // only the last two arguments were modified before this call within this function

      }
  }
}


__global__ void elim_dups(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
	  if(processinnextiteration[id])
	    {
	      unsigned srcc = cs.find(id);
	      unsigned dstc = partners[id];
	      
	      if(minwtcomponent[dstc] == eleminwts[id])
		{
		  if(id < goaheadnodeofcomponent[dstc])
		    {
		      processinnextiteration[id] = false;
		      //printf("duplicate!\n");
		    }
		}
	    }
	}
}

__global__ void dfindcompmin(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (inpid < graph.nnodes) id = inpid;

	if (id < graph.nnodes) {
	  if(partners[id] == graph.nnodes)
	    return;

	  unsigned srcboss = cs.find(id);
	  unsigned dstboss = cs.find(partners[id]);
	  if (id != partners[id] && srcboss != dstboss && eleminwts[id] != MYINFINITY && minwtcomponent[srcboss] == eleminwts[id] && dstboss != id && goaheadnodeofcomponent[srcboss] == id) {	// my edge is min outgoing-component edge.
	    if(!processinnextiteration[id]);
	      //printf("whoa!\n");
	    //= true;
	  }
	  else
	    {
	      if(processinnextiteration[id]);
		//printf("whoa2!\n");
	    }
	}
}

__global__ void dfindcompmintwo(unsigned *mstwt, Graph graph, ComponentSpace csw, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid, GlobalBarrier gb, bool *repeat, unsigned *count) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned id, nthreads = blockDim.x * gridDim.x;
	if (inpid < graph.nnodes) id = inpid;

	unsigned up = (graph.nnodes + nthreads - 1) / nthreads * nthreads;
	unsigned srcboss, dstboss;


	for(id = tid; id < up; id += nthreads) {
	  if(id < graph.nnodes && processinnextiteration[id])
	    {
	      srcboss = csw.find(id);
	      dstboss = csw.find(partners[id]);
	    }
	  
	  gb.Sync();
	  	  
	  if (id < graph.nnodes && processinnextiteration[id] && srcboss != dstboss) {
	    dprintf("trying unify id=%d (%d -> %d)\n", id, srcboss, dstboss);

	    if (csw.unify(srcboss, dstboss)) {
	      atomicAdd(mstwt, eleminwts[id]);
	      atomicAdd(count, 1);
	      dprintf("u %d -> %d (%d)\n", srcboss, dstboss, eleminwts[id]);
	      processinnextiteration[id] = false;
	      eleminwts[id] = MYINFINITY;	// mark end of processing to avoid getting repeated.
	    }
	    else {
	      *repeat = true;
	    }

	    dprintf("\tcomp[%d] = %d.\n", srcboss, csw.find(srcboss));
	  }

	  gb.Sync(); 
	}
}

int main(int argc, char *argv[]) {
  unsigned *mstwt, hmstwt = 0;
  int iteration = 0;
  Graph hgraph, graph;
  KernelConfig kconf;

  unsigned *partners, *phores;
  foru *eleminwts, *minwtcomponent;
  bool *processinnextiteration;
  unsigned *goaheadnodeofcomponent;
  const int nSM = kconf.getNumberOfSMs();

  double starttime, endtime;
  GlobalBarrierLifetime gb;
  const size_t compmintwo_res = maximum_residency(dfindcompmintwo, 384, 0);
  gb.Setup(nSM * compmintwo_res);

  if (argc != 2) {
    printf("Usage: %s <graph>\n", argv[0]);
    exit(1);
  }

  hgraph.read(argv[1]);
  hgraph.cudaCopy(graph);
  //graph.print();

  kconf.setProblemSize(graph.nnodes);
  ComponentSpace cs(graph.nnodes);

  if (cudaMalloc((void **)&mstwt, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating mstwt failed");
  CUDA_SAFE_CALL(cudaMemcpy(mstwt, &hmstwt, sizeof(hmstwt), cudaMemcpyHostToDevice));	// mstwt = 0.

  if (cudaMalloc((void **)&eleminwts, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating eleminwts failed");
  if (cudaMalloc((void **)&minwtcomponent, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating minwtcomponent failed");
  if (cudaMalloc((void **)&partners, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating partners failed");
  if (cudaMalloc((void **)&phores, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating phores failed");
  if (cudaMalloc((void **)&processinnextiteration, graph.nnodes * sizeof(bool)) != cudaSuccess) CudaTest("allocating processinnextiteration failed");
  if (cudaMalloc((void **)&goaheadnodeofcomponent, graph.nnodes * sizeof(unsigned)) != cudaSuccess) CudaTest("allocating goaheadnodeofcomponent failed");

  kconf.setMaxThreadsPerBlock();

  unsigned prevncomponents, currncomponents = graph.nnodes;

  bool repeat = false, *grepeat;
  CUDA_SAFE_CALL(cudaMalloc(&grepeat, sizeof(bool) * 1));
  CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));

  unsigned edgecount = 0, *gedgecount;
  CUDA_SAFE_CALL(cudaMalloc(&gedgecount, sizeof(unsigned) * 1));
  CUDA_SAFE_CALL(cudaMemcpy(gedgecount, &edgecount, sizeof(unsigned) * 1, cudaMemcpyHostToDevice));

  printf("finding mst.\n");
  starttime = rtclock();

  do {
    ++iteration;
    prevncomponents = currncomponents;
    dinit 		<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    //printf("0 %d\n", cs.numberOfComponentsHost());
    CudaTest("dinit failed");

cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    dfindelemin 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
struct timespec t1,t2;
	CudaTest("dfindelemin failed");
clock_gettime(CLOCK_MONOTONIC,&t1);
dfindelemin2        <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph.nnodes,graph.nedges,graph.noutgoing,graph.nincoming,graph.edgessrcdst,graph.edgessrcwt,graph.srcsrc,graph.psrc, 
 cs.ele2comp, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
cudaDeviceSynchronize();
clock_gettime(CLOCK_MONOTONIC,&t2);
CudaTest("dfindelemin2 failed");
cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 4096);
cudaDeviceSynchronize();
clock_gettime(CLOCK_MONOTONIC,&t1);

//#include "callVerify.cu"
//Added contents of file in callVerify.cu   
 verify_min_elem 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph.nnodes,graph.nedges,graph.noutgoing,graph.nincoming,graph.edgessrcdst,graph.edgessrcwt,graph.srcsrc,graph.psrc,
 cs.ele2comp, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);


cudaDeviceSynchronize();
clock_gettime(CLOCK_MONOTONIC,&t2);
printf("kernel verify time %fs\n",t2.tv_sec-t1.tv_sec+(t2.tv_nsec-t1.tv_nsec)/1e9);
    
//elim_dups 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    CudaTest("dfindelemin failed");
    if(debug) print_comp_mins(cs, graph, minwtcomponent, goaheadnodeofcomponent, partners, processinnextiteration);

    //printf("1 %d\n", cs.numberOfComponentsHost());
    //dfindcompmin 	<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes);
    //CudaTest("dfindcompmin failed");
    //printf("2 %d\n", cs.numberOfComponentsHost());
    //cudaThreadSynchronize();
    //printf("\n");
    //cs.copy(cs2);

    // if(debug) {
    //   cs.dump_to_file("components.txt");

    //   for(int i = 0; i < cs.nelements; i++) {
    // 	dfindcompmintwo_serial <<<1, 1>>> (mstwt, graph, cs2, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, i, gb, grepeat, gedgecount);
    //   }

    do {
      repeat = false;

      CUDA_SAFE_CALL(cudaMemcpy(grepeat, &repeat, sizeof(bool) * 1, cudaMemcpyHostToDevice));
      dfindcompmintwo <<<nSM * compmintwo_res, 384>>> (mstwt, graph, cs, eleminwts, minwtcomponent, partners, phores, processinnextiteration, goaheadnodeofcomponent, graph.nnodes, gb, grepeat, gedgecount);
      CudaTest("dfindcompmintwo failed");
		  
      CUDA_SAFE_CALL(cudaMemcpy(&repeat, grepeat, sizeof(bool) * 1, cudaMemcpyDeviceToHost));
    } while (repeat); // only required for quicker convergence?

    currncomponents = cs.numberOfComponentsHost();
    CUDA_SAFE_CALL(cudaMemcpy(&hmstwt, mstwt, sizeof(hmstwt), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&edgecount, gedgecount, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost));
    printf("\titeration %d, number of components = %d (%d), mstwt = %u mstedges = %u\n", iteration, currncomponents, prevncomponents, hmstwt, edgecount);
  } while (currncomponents != prevncomponents);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  endtime = rtclock();
	
  printf("\tmstwt = %u, iterations = %d.\n", hmstwt, iteration);
  printf("\t%s result: weight: %u, components: %u, edges: %u\n", argv[1], hmstwt, currncomponents, edgecount);
  printf("\truntime [mst] = %f ms.\n", 1000 * (endtime - starttime));

  // cleanup left to the OS.

  return 0;
}
