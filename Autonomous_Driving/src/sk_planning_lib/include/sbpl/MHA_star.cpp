/******************************************************************************************
*                                                                                        *
*    Yet Another Graph-Search Based Planning Library (YAGSBPL)                           *
*    A template-based C++ library for graph search and planning                          *
*    Version 2.1                                                                         *
*    ----------------------------------------------------------                          *
*    Copyright (C) 2013  Subhrajit Bhattacharya                                          *
*                                                                                        *
*    This program is free software: you can redistribute it and/or modify                *
*    it under the terms of the GNU General Public License as published by                *
*    the Free Software Foundation, either version 3 of the License, or                   *
*    (at your option) any later version.                                                 *
*                                                                                        *
*    This program is distributed in the hope that it will be useful,                     *
*    but WITHOUT ANY WARRANTY; without even the implied warranty of                      *
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                       *
*    GNU General Public License for more details <http://www.gnu.org/licenses/>.         *
*                                                                                        *
*                                                                                        *
*    Contact: subhrajit@gmail.com, http://subhrajit.net/                                 *
*                                                                                        *
*                                                                                        *
******************************************************************************************/
//    For a detailed tutorial and download, visit 
//    http://subhrajit.net/index.php?WPage=yagsbpl


template <class NodeType, class CostType>
void A_star_planner<NodeType,CostType>::init( GenericSearchGraphDescriptor<NodeType,CostType>* theEnv_p, bool createHashAndHeap )
{
	GraphNode_p thisGraphNode;

	if (theEnv_p && createHashAndHeap)
		GenericPlannerInstance.init(*theEnv_p, heapKeyCount, numberOfHeuristic);  // This initiates the graph, hash and heap of the generic planner
	else if (theEnv_p)
	    *GenericPlannerInstance.GraphDescriptor = *theEnv_p;
	
	// Remapping for coding convenience
	GraphDescriptor = GenericPlannerInstance.GraphDescriptor;
	hash = GenericPlannerInstance.hash;
	heap = GenericPlannerInstance.heap;
	
	// Init graph, clear the heap and clear stored paths just in case they not empty due to a previous planning
	GraphDescriptor->init();
	for( int j = 0; j < heap.size(); j++ ) heap[j]->clear();
	for( unsigned int j = 0; j < heap.size(); j++ ) {
		heap[j]->setId(j);
	}
	bpSgoal = NULL;

	for (int a=0; a<GraphDescriptor->SeedNodes.size(); a++)
	{
		// Check if node is in hash table - get it if it is, otherwise create it.
		thisGraphNode = hash->getNodeInHash( GraphDescriptor->SeedNodes[a] );
		
		// If node was created, initialize it
		if ( !thisGraphNode->initiated )
		{
			thisGraphNode->m_f = _heapFun(thisGraphNode->n, (CostType)0.0, GraphDescriptor->_getHeuristicsToTarget( thisGraphNode->n ), a);
            thisGraphNode->m_g = (CostType)0.0;
            thisGraphNode->m_h = GraphDescriptor->_getHeuristicsToTarget( thisGraphNode->n );
			thisGraphNode->came_from = NULL;
			thisGraphNode->plannerVars.seedLineage = a;
			thisGraphNode->plannerVars.g = (CostType)0.0;
			thisGraphNode->plannerVars.expanded0 = false;
			thisGraphNode->plannerVars.expandedi = false;
			
			if ( !GraphDescriptor->_isAccessible( thisGraphNode->n ) )
			{
				printf("ERROR (A_star): At least one of the seed nodes is not accessible!" );
				exit(1);
			}
			else
				thisGraphNode->plannerVars.accessible = true;
				
			thisGraphNode->initiated = true; // Always set this when other variables have already been set
		}
		
		// Push in heap
		for(unsigned int j = 0; j < heap.size(); j++ )
			heap[j]->push( thisGraphNode );
	}
}

// -----------------------------

template <class NodeType, class CostType>
void A_star_planner<NodeType,CostType>::clearLastPlanAndInit( GenericSearchGraphDescriptor<NodeType,CostType>* theEnv_p )
{
	for( int j = 0; j < heap.size(); j++ ) heap[j]->clear();
	heap.clear();
	heap.resize(numberOfHeuristic);
	bpSgoal = NULL;
	        
	// Set every node in hash to not expanded  
	if (hash && hash->HashTable) {
		for (int a=0; a<hash->hashTableSize; a++)
			for (int b=0; b<hash->HashTable[a].size(); b++)
			{
				hash->HashTable[a][b]->plannerVars.expanded0 = false;
				hash->HashTable[a][b]->plannerVars.expandedi = false;
				hash->HashTable[a][b]->initiated = false;
			}
	    // Clear the last plan, but not the hash table
	    if (theEnv_p)
		    init(theEnv_p, false);
	    else
		    init(GraphDescriptor, false);
    }
    else {
        if (theEnv_p)
		    init(theEnv_p);
	    else
		    init(GraphDescriptor);
	}
}

template <class NodeType, class CostType>
void A_star_planner<NodeType,CostType>::reInit(void)
{
	for( int j = 0; j < heap.size(); j++ ) heap[j]->clear();
	heap.clear();
	heap.resize(numberOfHeuristic);
	bpSgoal = NULL;
	        
	// Set every node in hash to not expanded  
	if (hash && hash->HashTable) {
		for (int a=0; a<hash->hashTableSize; a++)
			for (int b=0; b<hash->HashTable[a].size(); b++) {
					hash->HashTable[a][b]->plannerVars.expanded0 = false;
					hash->HashTable[a][b]->plannerVars.expandedi = false;
					hash->HashTable[a][b]->m_f = _heapFun(hash->HashTable[a][b]->n, hash->HashTable[a][b]->plannerVars.g, 
					                                      GraphDescriptor->_getHeuristicsToTarget( hash->HashTable[a][b]->n ), 
					                                      hash->HashTable[a][b]->plannerVars.seedLineage);
                    hash->HashTable[a][b]->m_g = hash->HashTable[a][b]->plannerVars.g;
					hash->HashTable[a][b]->m_h = GraphDescriptor->_getHeuristicsToTarget( hash->HashTable[a][b]->n ); 
					for(unsigned int j = 0; j < heap.size(); j++ )
						heap[j]->push(hash->HashTable[a][b]);
			}
	}
}

// ==================================================================================

template <class NodeType, class CostType>
std::vector<NodeType> A_star_planner<NodeType,CostType>::getHashTableAll(void)
{
	std::vector< NodeType > ret;
	ret.clear();
	if (hash && hash->HashTable)
	{
		for (int a=0; a<hash->hashTableSize; a++)
			for (int b=0; b<hash->HashTable[a].size(); b++)
			{
				ret.push_back(hash->HashTable[a][b]->n);
			}
	}
	return (ret);
}

template <class NodeType, class CostType>
std::vector<NodeType> A_star_planner<NodeType,CostType>::getHashTableExpanded(void)
{
	std::vector< NodeType > ret;
	ret.clear();
	if (hash && hash->HashTable)
	{
		for (int a=0; a<hash->hashTableSize; a++)
			for (int b=0; b<hash->HashTable[a].size(); b++)
			{
				if( hash->HashTable[a][b]->expandedi )
					ret.push_back(hash->HashTable[a][b]->n);
			}
	}
	return (ret);
}

template <class NodeType, class CostType>
std::vector<NodeType> A_star_planner<NodeType,CostType>::getHashTableNotExpanded(void)
{
	std::vector< NodeType > ret;
	ret.clear();
	if (hash && hash->HashTable)
	{
		for (int a=0; a<hash->hashTableSize; a++)
			for (int b=0; b<hash->HashTable[a].size(); b++)
			{
				if( !(hash->HashTable[a][b]->expandedi) )
					ret.push_back(hash->HashTable[a][b]->n);
			}
	}
	return (ret);
}

// ==================================================================================

template <class NodeType, class CostType>
std::vector<CostType> A_star_planner<NodeType,CostType>::_heapFun(NodeType& n, CostType g, std::vector<CostType> h, int s) 
{
    if (heapFun_fp)
        return ( heapFun_fp(n, g, h, s) );

	std::vector<CostType> ret;
	ret.resize(h.size());
	for( unsigned int j = 0; j < h.size(); j++ )
		ret[j] = g + w1*h[j];
	return (ret);
}

// ==================================================================================
template <class NodeType, class CostType>
void A_star_planner<NodeType,CostType>::switchHeap()
{
    int bestHF(0);
#if 1
    bestHF = (currentSearchHF+1)%numberOfHeuristic;
    while( !activeMHA[bestHF] )
    {
        bestHF = (bestHF+1)%numberOfHeuristic;
    }
#elif 0
    const int prevSearchHF(currentSearchHF);
    double bestC(std::numeric_limits<CostType>::infinity());
    if( prevSearchHF )
    {
        bestC = heap[bestHF]->heapArray[0]->gh(bestHF);
    }
    for(int j = 1; j < numberOfHeuristic; j++)
    {
        if( j == prevSearchHF )
            continue;
        if( !activeMHA[j] )
            continue;
        if( bestC > heap[j]->heapArray[0]->gh(j) )
        {
            bestC = heap[j]->heapArray[0]->gh(j);
            bestHF = j;
        }
    }
#elif 0
    const int prevSearchHF(currentSearchHF);
    double bestC(heap[prevSearchHF]->heapArray[0]->gh(prevSearchHF));
    if( !prevSearchHF )
    {
	    bestC = std::numeric_limits<CostType>::infinity();
    }
    for(int j = 1; j < numberOfHeuristic; j++)
    {
        if( j == prevSearchHF )
            continue;
        if( !activeMHA[j] )
            continue;
        if( bestC > heap[j]->heapArray[0]->gh(j) )
        {
            bestC = heap[j]->heapArray[0]->gh(j);
            bestHF = j;
        }
    }
	//if( bestHF == currentSearchHF )
	//	bestHF = 0;
#elif 0
    const int prevSearchHF(currentSearchHF);
    double bestC(heap[prevSearchHF]->heapArray[0]->h(prevSearchHF));
    if( !prevSearchHF )
    {
	    bestC = std::numeric_limits<CostType>::infinity();
    }
    for(int j = 1; j < numberOfHeuristic; j++)
    {
        if( j == prevSearchHF )
            continue;
        if( !activeMHA[j] )
            continue;
        if( bestC > heap[j]->heapArray[0]->h(j) )
        {
            bestC = heap[j]->heapArray[0]->h(j);
            bestHF = j;
        }
    }
	//if( bestHF == currentSearchHF )
	//	bestHF = 0;
#elif 0
    const int prevSearchHF(currentSearchHF);
    double bestC(std::numeric_limits<CostType>::infinity());
    if( prevSearchHF )
    {
        bestC = heap[bestHF]->heapArray[0]->h(bestHF);
    }
    for(int j = 1; j < numberOfHeuristic; j++)
    {
        if( j == prevSearchHF )
            continue;
        if( !activeMHA[j] )
            continue;
        if( bestC > heap[j]->heapArray[0]->h(j) )
        {
            bestC = heap[j]->heapArray[0]->h(j);
            bestHF = j;
        }
    }
#else
    const int prevSearchHF(currentSearchHF);
    double bestC(std::numeric_limits<CostType>::infinity());
    if( prevSearchHF )
    {
        bestC = heap[bestHF]->heapArray[0]->gh(bestHF);
    }
    for(int j = 1; j < numberOfHeuristic; j++)
    {
        if( j == prevSearchHF )
            continue;
        if( !activeMHA[j] )
            continue;
        if( bestC > heap[j]->heapArray[0]->gh(j) )
        //if( !(bestC < heap[j]->heapArray[0]->gh(j)) )
        {
            bestC = heap[j]->heapArray[0]->gh(j);
            bestHF = j;
        }
    }
#endif
    currentSearchHF = bestHF;
}

template <class NodeType, class CostType>
void A_star_planner<NodeType,CostType>::plan(std::vector<unsigned int>* expands)
{
	GraphNode_p thisGraphNode;//, thisNeighbourGraphNode;
	unsigned int i; // MHA*
	gSgoal = std::numeric_limits<CostType>::infinity(); // MHA* L17
	bpSgoal = NULL;
    currentSearchHF = 0;
    lastSearchCost = std::numeric_limits<CostType>::infinity();
    lastReferenceCost = 0.0; //std::numeric_limits<CostType>::infinity();
    prevReferenceCost = 0.0; //std::numeric_limits<CostType>::infinity();
    unsigned int prevSearchHF(0);
	
	#if _YAGSBPL_A_STAR__VIEW_PROGRESS
		float timediff = 0.0;
		expandcount = 0;
		expandCount0 = 0;
		expandCounti = 0;
		startclock = clock();
		startsecond = time(NULL);
	#endif
	while ( !heap[0]->empty() ) // MHA* L22
	{
		#if _YAGSBPL_A_STAR__VIEW_PROGRESS
		    if(ProgressShowInterval>0) {
			    if (expandcount % ProgressShowInterval == 0)
			    {
				    if (timediff>=0.0)
					    timediff = ((float)(clock()-startclock)) / ((float)CLOCKS_PER_SEC);
				    printf("Number of states expanded: %d. Heap size: %d. Time elapsed: %f s.\n", 
						    expandcount, heap[0]->size(), ((timediff>=0.0) ? timediff : difftime(time(NULL),startsecond)) );
			    }
			}
		#endif
		if(numberOfHeuristic < 2 ) { // Single Heuristic
			if( bpSgoal ) {
				#if _YAGSBPL_A_STAR__VIEW_PROGRESS
				    timediff = ((float)(clock()-startclock)) / ((float)CLOCKS_PER_SEC);
				    printf("Number of states expanded: %d. Heap size: %d. Time elapsed: %f s. Number of heuristic: %d\n", 
					    expandcount, heap[0]->size(), ((timediff>=0.0) ? timediff : difftime(time(NULL),startsecond)), numberOfHeuristic);
					printf("We expand %u for admissible heuristic and %u for additional heuristics.\n", expandCount0, expandCounti);
				#endif
                if( expands )
                {
                    expands->clear();
                    expands->push_back(expandCount0);
                    expands->push_back(expandCounti);
                    printf("[Single] expands->(%u,%u) = (%u,%u).\n", expands->at(0), expands->at(1), expandCount0, expandCounti);
                }
				return;
			}
			if( !heap[0]->empty() ) {
				thisGraphNode = heap[0]->pop();
				thisGraphNode->plannerVars.expanded0 = true; // Put in closed list
				expand(thisGraphNode,0); // MHA* L28
			}
		}
		else { // Multi Heuristic
#if 1
            // check if we stuck by local minima
            bool changeHeap(heap[currentSearchHF]->empty());
            if( !changeHeap )
            {
                changeHeap = heap[currentSearchHF]->heapArray[0]->gh(currentSearchHF) > lastSearchCost + 1.0e-6;
            }
            if( !changeHeap )
            {
                changeHeap = heap[currentSearchHF]->heapArray[0]->gh(currentSearchHF) > w2*(heap[0]->heapArray[0]->gh(0));
            }
            if( changeHeap )
            //if( heap[currentSearchHF]->heapArray[0]->gh(currentSearchHF) > lastSearchCost + 1.0e-6 )
            {
                // save current reference cost if current heap is 0
                if( currentSearchHF == 0 )
                    lastReferenceCost = heap[0]->heapArray[0]->f(0);
                else
                {
                    prevReferenceCost = heap[currentSearchHF]->heapArray[0]->f(currentSearchHF);
                    prevSearchHF = currentSearchHF;
                }
#if _MHAs_VIEW_HEURISTIC_CHANGE
                printf("[MHA*] : Switch heuristic from %d to ", currentSearchHF);
#endif
                switchHeap();
#if _MHAs_VIEW_HEURISTIC_CHANGE
                printf("%d.\n", currentSearchHF);
#endif
            }
            // return to heap 0
            if( currentSearchHF )
            {
                if( heap[0]->heapArray[0]->f(0) < lastReferenceCost - 1.0e-6)//*w2 )
                    currentSearchHF = 0;
                //else if( heap[prevSearchHF]->heapArray[0]->f(prevSearchHF) < prevReferenceCost - 1.0e-6*w2 )
                //{
                //    currentSearchHF = prevSearchHF;
                //    prevReferenceCost = -10.0;
                //    prevSearchHF = 0;
                //}
                //else
                //    lastReferenceCost = heap[0]->heapArray[0]->gh(0);
            }
            // expand the node with currentSearchHF
           	//if( gSgoal < heap[currentSearchHF]->heapArray[0]->f(currentSearchHF) ) {// MHA* L25
           	if( gSgoal < w2*heap[0]->heapArray[0]->gh(0) ) {// MHA* L25
           	//if( gSgoal < w2*heap[0]->heapArray[0]->f(0) ) {// MHA* L25
				//bookmarkGraphNodes.push_back(thisGraphNode); // MHA* L26
				#if _YAGSBPL_A_STAR__VIEW_PROGRESS
				    timediff = ((float)(clock()-startclock)) / ((float)CLOCKS_PER_SEC);
				    printf("Number of states expanded: %d. Heap size: %d. Time elapsed: %f s. Number of heuristic: %d\n", 
					    expandcount, heap[0]->size(), ((timediff>=0.0) ? timediff : difftime(time(NULL),startsecond)), numberOfHeuristic);
						printf("We expand %u for admissible heuristic and %u for additional heuristics.\n", expandCount0, expandCounti);
				#endif
                /* if( expands )
                {
                    expands->clear();
                    expands->push_back(expandCount0);
                    expands->push_back(expandCounti);
                    printf("[Multi] expands->(%u,%u) = (%u,%u).\n", expands->at(0), expands->at(1), expandCount0, expandCounti);
				}*/
				//ROS_INFO("Return 0, gSgoal = %.1f, heap[0]->heapArray[0]->gh(0) = %.1f.", gSgoal, heap[0]->heapArray[0]->gh(0));
				return; // MHA* L26
			}
			thisGraphNode = heap[currentSearchHF]->pop(); // MHA* L27
            lastSearchCost = thisGraphNode->gh(currentSearchHF);
			//thisGraphNode->plannerVars.expandedi = true; // Put in closed list
			expand(thisGraphNode,currentSearchHF); // MHA* L28
#else
			for( i = 1; i < numberOfHeuristic; i++ ) { // MHA* L23
				if( !heap[i]->empty() ) {
					if( activeMHA[i] && (heap[i]->heapArray[0]->f(i) <= w2*heap[0]->heapArray[0]->f(0)) ) // MHA* L24
					{
                    	if( gSgoal < heap[i]->heapArray[0]->f(i) ) {// MHA* L25
							//bookmarkGraphNodes.push_back(thisGraphNode); // MHA* L26
							#if _YAGSBPL_A_STAR__VIEW_PROGRESS
							    timediff = ((float)(clock()-startclock)) / ((float)CLOCKS_PER_SEC);
							    printf("Number of states expanded: %d. Heap size: %d. Time elapsed: %f s. Number of heuristic: %d\n", 
								    expandcount, heap[0]->size(), ((timediff>=0.0) ? timediff : difftime(time(NULL),startsecond)), numberOfHeuristic);
								printf("We expand %u for admissible heuristic and %u for additional heuristics.\n", expandCount0, expandCounti);
							#endif
							ROS_INFO("Return 0, gSgoal = %.1f", gSgoal);
							return; // MHA* L26
						}
						thisGraphNode = heap[i]->pop(); // MHA* L27
						thisGraphNode->plannerVars.expandedi = true; // Put in closed list
						expand(thisGraphNode,i); // MHA* L28
					}
					else // MHA* L29
                    {
						if( gSgoal < heap[0]->heapArray[0]->f(0) ) { // MHA* L30
							//bookmarkGraphNodes.push_back(thisGraphNode); // MHA* L31
							#if _YAGSBPL_A_STAR__VIEW_PROGRESS
							    timediff = ((float)(clock()-startclock)) / ((float)CLOCKS_PER_SEC);
							    printf("Number of states expanded: %d. Heap size: %d. Time elapsed: %f s. Number of heuristic: %d\n", 
								    expandcount, heap[0]->size(), ((timediff>=0.0) ? timediff : difftime(time(NULL),startsecond)), numberOfHeuristic);
								printf("We expand %u for admissible heuristic and %u for additional heuristics.\n", expandCount0, expandCounti);
							#endif
							ROS_INFO("Return 1");
							return; // MHA* L31
						}
						thisGraphNode = heap[0]->pop(); // MHA* L32
						thisGraphNode->plannerVars.expanded0 = true; // Put in closed list
						expand(thisGraphNode,0); // MHA* L33
					}
				}
			}
#endif
		}
	}
    gSgoal = NULL;
	printf("The heap is empty.\n");
}

template <class NodeType, class CostType>
void A_star_planner<NodeType,CostType>::expand(GraphNode_p& thisGraphNode, int id) // MHA* L3
{
	#if _YAGSBPL_A_STAR__VIEW_PROGRESS
	    expandcount++;
	    if( id > 0 )
	    	expandCounti++;
	    else
		    expandCount0++;
	#endif
	GraphNode_p thisNeighbourGraphNode;
	CostType this_g_val, thisTransitionCost, test_g_val;
	std::vector< NodeType > thisNeighbours;
	std::vector< CostType > thisTransitionCosts;
	int a;
	
//	ROS_INFO("expand(): Now we expand a node with g = %.3f, f[0] = %.3f.", thisGraphNode->plannerVars.g, thisGraphNode->m_f[0]);
	// before expand lets check if this node is th goal
	// As the algorithm considers the case of single goal, we need this procedure
	if( GraphDescriptor->_stopSearch( thisGraphNode->n ) ) {
		//ROS_INFO("expand(): This is a goal node. with g = %.f.",thisGraphNode->plannerVars.g);
		if( thisGraphNode->plannerVars.g < gSgoal ) {
			gSgoal = thisGraphNode->plannerVars.g;
			bpSgoal = thisGraphNode;
		}
	}
	
  	a = GraphDescriptor->_deactivateSearch( thisGraphNode->n );
  	if( a > 0 )
   	{
#if _MHAs_VIEW_HEURISTIC_CHANGE
   	    ROS_INFO("Deactivate MHA* with id = %d and we expanded addmissible/inaddmissible heuristics of %u/%u.", a, expandCount0, expandCounti);
#endif
   	    //activeMHA[a] = false;
#if 0
        for( unsigned int j = a+1; j < activeMHA.size(); j++ )
            activeMHA[j] = false;
#elif 1
		for( unsigned int j = 1; j <= a; j++ )
            activeMHA[j] = false;
#endif
 	}

	for( unsigned int i = 0; i < numberOfHeuristic; i++ ) {
		if( activeMHA[i] && thisGraphNode->inHeap[i] ) {
//			ROS_INFO("expand(): We remove this node from heap %u.", i);
			heap[i]->remove(thisGraphNode); // MHA* L4
		}
//		else
//			ROS_INFO("expand(): This node is already removed from heap %u.", i);
	}
	thisGraphNode->plannerVars.expanded0 = true; // Put in closed list
	thisGraphNode->plannerVars.expandedi = true; // Put in closed list
//	ROS_INFO("expand(): Now we removed this node from other heaps.");
	// get successors
	thisNeighbours.clear();
	thisTransitionCosts.clear();
	if ( thisGraphNode->successors.empty() ) // Successors were not generated previously
	{
		GraphDescriptor->_getSuccessors( thisGraphNode->n , &thisNeighbours , &thisTransitionCosts );
//		ROS_INFO("expand(): We get %zu successors.", thisNeighbours.size());
		thisGraphNode->successors.init( thisNeighbours.size() );
//		ROS_INFO("expand(): We initialize the successors.");
		for (a=0; a<thisNeighbours.size(); a++)
			thisGraphNode->successors.set(a, hash->getNodeInHash(thisNeighbours[a]), thisTransitionCosts[a]);
	}
//	ROS_INFO("expand(): This node has %zu == %zu == %d successors.", thisNeighbours.size(), thisTransitionCosts.size(), (thisGraphNode->successors).size());
	this_g_val = thisGraphNode->plannerVars.g;
	bool localMinima = true;
	int count0 = 0, counti = 0;
	for (a=0; a<thisGraphNode->successors.size(); a++) // MHA* L5
	{
		thisNeighbourGraphNode = thisGraphNode->successors.getLinkSearchGraphNode(a);
		thisTransitionCost = thisGraphNode->successors.getLinkCost(a);
		// An uninitiated neighbour node - definitely g & f values not set either.
		if ( !thisNeighbourGraphNode->initiated ) // MHA* L6
		{
//			ROS_INFO("expand(): Child %u is not initiated.", a);
			thisNeighbourGraphNode->plannerVars.accessible = GraphDescriptor->_isAccessible( thisNeighbourGraphNode->n );
			if ( thisNeighbourGraphNode->plannerVars.accessible )
			{
//				ROS_INFO("expand(): This child is accessible.");
				thisNeighbourGraphNode->came_from = NULL; // MHA* L7
				thisNeighbourGraphNode->plannerVars.seedLineage = thisGraphNode->plannerVars.seedLineage;
				thisNeighbourGraphNode->plannerVars.g = std::numeric_limits<CostType>::infinity(); // MHA* L7
				thisNeighbourGraphNode->m_f.resize(numberOfHeuristic,std::numeric_limits<CostType>::infinity());
				thisNeighbourGraphNode->m_h.resize(numberOfHeuristic,std::numeric_limits<CostType>::infinity());
				thisNeighbourGraphNode->inHeap.resize(numberOfHeuristic,false);
				thisNeighbourGraphNode->heapArrayPos.resize(numberOfHeuristic,-1);
//				for( unsigned int i = 0; i < numberOfHeuristic; i++ ) {
//					thisNeighbourGraphNode->m_f[i] = std::numeric_limits<CostType>::infinity(); // MHA* L7
//					thisNeighbourGraphNode->inHeap[i] = false; // MHA* L7
//					thisNeighbourGraphNode->heapArrayPos[i] = -1; // MHA* L7
//				}
				thisNeighbourGraphNode->plannerVars.expanded0 = false;
				thisNeighbourGraphNode->plannerVars.expandedi = false;
			}
			thisNeighbourGraphNode->initiated = true; // Always set this when other variables have already been set
//			continue;
		}
		if( thisNeighbourGraphNode->plannerVars.g > thisGraphNode->plannerVars.g + thisTransitionCost ) { // MHA* L8
//			ROS_INFO("expand(): We update the g value and parent of this child.");
			thisNeighbourGraphNode->plannerVars.g = thisGraphNode->plannerVars.g + thisTransitionCost; // MHA* L9
			thisNeighbourGraphNode->incSizeTo(numberOfHeuristic);
			thisNeighbourGraphNode->m_g = thisNeighbourGraphNode->plannerVars.g;
			thisNeighbourGraphNode->m_h = GraphDescriptor->_getHeuristicsToTarget( thisNeighbourGraphNode->n );
			thisNeighbourGraphNode->m_f = _heapFun(thisNeighbourGraphNode->n, thisNeighbourGraphNode->m_g, 
				                                      thisNeighbourGraphNode->m_h, 
				                                      thisGraphNode->plannerVars.seedLineage);
			/* thisNeighbourGraphNode->m_f = _heapFun(thisNeighbourGraphNode->n, thisNeighbourGraphNode->plannerVars.g, 
				                                      GraphDescriptor->_getHeuristicsToTarget( thisNeighbourGraphNode->n ), 
				                                      thisGraphNode->plannerVars.seedLineage);*/
			thisNeighbourGraphNode->came_from = thisGraphNode; // MHA* L9
			if( !thisNeighbourGraphNode->plannerVars.expanded0 ) { // MHA* L10
//				ROS_INFO("expand(): We add this child to heap 0.");
				heap[0]->push( thisNeighbourGraphNode ); // MHA* L11
				//count0++;
				if( !thisNeighbourGraphNode->plannerVars.expandedi ) { // MHA* L12
//					ROS_INFO("expand(): We add this child to heap i.");
					for( unsigned int i = 1; i < numberOfHeuristic; i++ ) { // MHA* L13
//						ROS_INFO("expand(): We compaer f(%u) = %.3f < %.3f * %.3f", i, thisNeighbourGraphNode->m_f[i], w2, thisNeighbourGraphNode->m_f[0]);
						//if( true || thisNeighbourGraphNode->f(i) < w2*thisNeighbourGraphNode->f(0) ) { // MHA* L14
						if( activeMHA[i] && thisNeighbourGraphNode->f(i) < w2*thisNeighbourGraphNode->f(0) ) { // MHA* L14
//							ROS_INFO("expand(): We add this child to heap %u.", i);
							heap[i]->push( thisNeighbourGraphNode ); // MHA* L15
							//counti++;
						}
					}
				}
				
			}
		}
	}
//	ROS_INFO("expand(): DONE. We add count0 = %d = counti = %d.", count0, counti);
//	isFeasibleSearch();
//	printf("expane f[0] = %.5f.\n",thisGraphNode->m_f[0]);
//	if( localMinima ) printf("This node is a local minima.\n");
//	return (localMinima);
//	ROS_INFO("expand(): DONE.");
}

// ==================================================================================

template <class NodeType, class CostType>
SearchGraphNode< NodeType, CostType, A_star_variables<CostType> >*
											A_star_planner<NodeType,CostType>::getGoalGraphNodePointer(void)
{
	return (bpSgoal);
}

template <class NodeType, class CostType>
NodeType A_star_planner<NodeType,CostType>::getGoalNode(void)
{
	return (bpSgoal->n);
}

template <class NodeType, class CostType>
CostType A_star_planner<NodeType,CostType>::getPlannedPathCost(void)
{
	return (bpSgoal->plannerVars.g);
}

template <class NodeType, class CostType>
std::vector< NodeType > A_star_planner<NodeType,CostType>::getPlannedPath(std::vector<double>* g /* = NULL*/)
{
	std::vector< NodeType > path;
	path.clear();
	if( g )
	    g->clear();
	// Reconstruct path
	GraphNode_p thisGraphNode = bpSgoal;
	while (thisGraphNode)
	{
		path.push_back(thisGraphNode->n);
		if( g )
		    g->push_back(thisGraphNode->plannerVars.g);
		thisGraphNode = thisGraphNode->came_from;
	}
	return (path);
}

template <class NodeType, class CostType>
A_star_variables<CostType> A_star_planner<NodeType,CostType>::getNodeInfo(NodeType n)
{
	GraphNode_p thisGraphNode = hash->getNodeInHash(n);
	return thisGraphNode->plannerVars;
}

