/******************************************************************************************
*                                                                                        *
*    Yet Another Graph-Search Based Planning Library (YAGSBPL)                           *
*    A template-based C++ library for graph search and planning                          *
*    Version 2.0                                                                         *
*    ----------------------------------------------------------                          *
*    Copyright (C) 2011  Subhrajit Bhattacharya                                          *
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


#ifndef __A_STAR_2F585H2B321R_H_
#define __A_STAR_2F585H2B321R_H_


#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "yagsbplMHA_base.h"

#define _YAGSBPL_A_STAR__VIEW_PROGRESS 0
#define _YAGSBPL_A_STAR__PRINT_VERBOSE 0 // requires 'void print(std::string pre)' as a member function of the node
#define _YAGSBPL_A_STAR__HANDLE_EVENTS 0
#define _MHAs_TEST_NEW_CODE 1
#define _MHAs_VIEW_HEURISTIC_CHANGE 0

template <class CostType>
class A_star_variables
{
public:
	CostType g;
	bool expanded0; // Whether in closed list or not in Heap0
	bool expandedi; // Whether in closed list or not in other Heaps
	bool accessible; // Since the environment is assumed to to change, each node has fixed accessibility
	int seedLineage; // stores which seed the node came from
	
	A_star_variables() { expanded0=false; expandedi=false; seedLineage=-1; }
};

template <class NodeType, class CostType>
class A_star_planner
{
public:
	// typedef's for convenience:
	typedef  A_star_variables<CostType>  PlannerSpecificVariables;
	typedef  SearchGraphNode< NodeType, CostType, PlannerSpecificVariables >*  GraphNode_p;
	
	// Instance of generac planner
	GenericPlanner< NodeType, CostType, PlannerSpecificVariables > GenericPlannerInstance;
	// Re-mapping of generic planner variables for ease of use (coding convenience)
	GenericSearchGraphDescriptor<NodeType,CostType>* GraphDescriptor;
	HashTableContainer<NodeType,CostType,PlannerSpecificVariables>* hash;
	std::vector< HeapContainer<NodeType,CostType,PlannerSpecificVariables>* > heap; // MHA*
	
	// Member variables
	int heapKeyCount;
	int ProgressShowInterval;
//	std::vector< GraphNode_p > bookmarkGraphNodes;
	unsigned int numberOfHeuristic; // MHA*
	double w1, w2; // MHA*
	CostType gSgoal; // MHA*
	GraphNode_p bpSgoal; // MHA*
	std::vector<bool> activeMHA; // MHA*
    int currentSearchHF; // MHA*
    CostType lastSearchCost; // MHA*
    CostType lastReferenceCost; // MHA*
    CostType prevReferenceCost; // MHA*
	GraphNode_p openTop(unsigned int i);

	// Optional event handlers - Pointers to function that get called when an event take place
	#if _YAGSBPL_A_STAR__HANDLE_EVENTS
		// Node 'n' is expanded. This can also be handeled by 'stopSearch'.
		void (*event_NodeExpanded_g)(NodeType n, CostType gVal, CostType fVal, int seedLineage);
		void (NodeType::*event_NodeExpanded_nm)(CostType gVal, CostType fVal, int seedLineage);
		// Successor 'nn' is in open list and is just initiated or is updated
		void (*event_SuccUpdated_g)(NodeType n, NodeType nn, CostType edgeCost, CostType gVal, CostType fVal, int seedLineage);
		void (NodeType::*event_SuccUpdated_nm)(NodeType nn, CostType edgeCost, CostType gVal, CostType fVal, int seedLineage);
	#endif
	
	// Helper functions for specialized applications
	// Computes the f-value:
	std::vector<CostType> (*heapFun_fp)(NodeType& n, CostType g, std::vector<CostType> h, int s);
	std::vector<CostType> _heapFun(NodeType& n, CostType g, std::vector<CostType> h, int s);
	
	// Initializer and planner
	A_star_planner()
		{ hash = NULL; heapKeyCount = 200; ProgressShowInterval = 10000; 
		  //event_NodeExpanded_g=NULL; event_NodeExpanded_nm=NULL; event_SuccUpdated_g=NULL; event_SuccUpdated_nm=NULL; 
		  heapFun_fp = NULL;
		  numberOfHeuristic = 1; w1 = 1.0; w2 = 1.0; /*MHA**/ }
	void setParams( double eps=1.0, int heapKeyCt=200, int progressDispInterval=10000 ) // call to this is optional.
		{ w1 = eps; heapKeyCount = heapKeyCt; ProgressShowInterval = progressDispInterval; }
	void setMaxNumberOfHeuristicFunction( const unsigned int& nh ) { numberOfHeuristic = nh; activeMHA.resize(nh,true); }; // MHA*
	void setSuboptimality( double eps1, double eps2 ) { w1 = eps1; w2 = eps2; }
	void init( GenericSearchGraphDescriptor<NodeType,CostType>* theEnv_p=NULL , bool createHashAndHeap=true );
	void init( GenericSearchGraphDescriptor<NodeType,CostType> theEnv ) { init( &theEnv); } // for version compatability
	void reInit(void);
	void plan(std::vector<unsigned int>* expands = NULL);
    void switchHeap();
	
	// Functions to get node expanded
	std::vector<NodeType> getHashTableAll(void);
	std::vector<NodeType> getHashTableExpanded(void);
	std::vector<NodeType> getHashTableNotExpanded(void);

	// Clear the last plan, but not the hash table. Must be called after at least one initialization.
	//   Use this if you intend to re-plan with different start/goal, 
	//   and/or if isAccessible is changed such that the new planning is being done in a sub-graph of the previous graph,
	//   Won't give correct result if cost function has changed or the new graph has new edges attached to nodes explored in old graph.
	void clearLastPlanAndInit( GenericSearchGraphDescriptor<NodeType,CostType>* theEnv_p=NULL );
	
	// Planner output access: ( to be called after plan(), and before destruction of planner )
	NodeType getGoalNode(void);
	GraphNode_p getGoalGraphNodePointer(void);
	std::vector< NodeType > getPlannedPath(std::vector<double>* g = NULL);
	CostType getPlannedPathCost(void);
	A_star_variables<CostType> getNodeInfo(NodeType n);
	
	// expand state
	void expand(GraphNode_p& thisGraphNode, int id);
	// TODO:Add new Heuristics
	//void addNewHeuristics(GraphNode_p& thisGraphNode);
	// check feasibility of current search
	//bool isFeasibleSearch();
	
	// Other variables for querying progress of planning process from other functions
	unsigned int expandCount0, expandCounti;
	#if  _YAGSBPL_A_STAR__VIEW_PROGRESS
		clock_t  startclock;
		time_t startsecond;
		int expandcount;
	#endif
};

// ------------------------------------------------------------------------------
// ------------------------------------------------------------------------------
// Since we use templates, the definitions need to be included in the header file as well.

#include "MHA_star.cpp"

#endif

