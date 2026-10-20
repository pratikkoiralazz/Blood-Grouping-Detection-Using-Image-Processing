import numpy as np

def genetic_algorithm_feature_selection(X, y, generations=5, pop_size=10):
    num_features = X.shape[1]
    # Initialize random binary population (masks for features)
    population = np.random.randint(2, size=(pop_size, num_features))
    
    best_mask = population[0]
    
    for gen in range(generations):
        scores = []
        for ind in population:
            selected_idx = np.where(ind == 1)[0]
            if len(selected_idx) == 0:
                scores.append(0)
                continue
            # Simple fitness proxy using correlation / variance
            score = np.sum(np.var(X[:, selected_idx], axis=0))
            scores.append(score)
            
        best_idx = np.argmax(scores)
        best_mask = population[best_idx]
        
        # Simple crossover & mutation placeholder
        population = np.random.randint(2, size=(pop_size, num_features))
        population[0] = best_mask # Elitism
        
    return best_mask