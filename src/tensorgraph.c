#include <lib/tensorgraph.h>

void free_tensorgraph(TensorGraph* graph){
	if(!graph) return;
	if(graph->items) free(graph->items);
	free(graph);
}

void search_tensors(TensorGraph* graph, Tensor* tensor){
	if(!graph || !tensor || tensor->visited == -1) return;
	tensor->visited = 1;
	static int depth = 0;
	depth += 1;

	if(tensor->creator){
		search_tensors(graph, tensor->creator->input1);
		search_tensors(graph, tensor->creator->input2);
	}

	add_to_tensorgraph(graph, tensor);
}

int resize_tensorgraph(TensorGraph* graph){
	if(!graph || !graph->items) return 0;
	if(graph->capacity == 0) graph->capacity = INITIAL_CAPACITY;
	int new_capacity = 2 * graph->capacity;

	Tensor** new_items = (Tensor**)realloc(graph->items, new_capacity * sizeof(Tensor*));
	if(!new_items) return 0;

	graph->items = new_items;
	graph->capacity = new_capacity;
	return 1;
}

int unvisit_tensors(TensorGraph* graph){
	if(!graph || !graph->items) return 0;

	for(int i = 0; i < graph->size; i++){
		if(!graph->items[i]) return i;
		graph->items[i]->visited = 0;
	}

	return graph->size;
}

int add_to_tensorgraph(TensorGraph* graph, Tensor* tensor){
	if(!graph || !tensor) return 0;
	if(graph->size >= graph->capacity){
		int resized = resize_tensorgraph(graph);
		if(!resized) return 0;
	}

	graph->items[graph->size] = tensor;
	graph->size++;
	return 1;
}

TensorGraph* create_tensorgraph(Tensor* root){
	if(!root) return NULL;

	TensorGraph* graph = (TensorGraph*)malloc(sizeof(TensorGraph));
	if(!graph) return NULL;
	
	int initial_capacity = INITIAL_CAPACITY;
	graph->items = (Tensor**)malloc(initial_capacity * sizeof(Tensor*));
	if(!graph->items){
		free(graph);
		return NULL;
	}
	graph->capacity = initial_capacity;
	graph->size = 0;

	search_tensors(graph, root);
	(void)unvisit_tensors(graph);

	return graph;
}

void calculate_gradients(TensorGraph* graph){
	if(!graph || !graph->items) return;
	for(int i = graph->size-1; i > -1; i--){
		Tensor* t = graph->items[i];
		if(t && t->creator) t->creator->back(t);
	}
}

void zero_gradients(TensorGraph* graph){
	if(!graph || !graph->items) return;
	for(int i = 0; i < graph->size-1; i++){
		Tensor* t = graph->items[i];
		if(t && t->creator) zero_gradient(t);
	}
}
