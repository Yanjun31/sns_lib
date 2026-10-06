import torch


def k_means(X:torch.Tensor, num_clusters:int, max_iters:int=100) ->tuple[torch.Tensor, torch.Tensor]:
    # Move dataset to target device (e.g., 'cuda' or 'cpu')
    num_samples, _ = X.shape
    
    # 1. Randomly initialize centroids from the dataset points
    random_indices = torch.randperm(num_samples)[:num_clusters]
    centroids = X[random_indices]
    
    for i in range(max_iters):
        # 2. Compute Euclidean distance from each point to each centroid
        # Formula: (X - centroids)^2 = X^2 - 2*X*centroids^T + centroids^2
        distances = torch.cdist(X, centroids, p=2)
        
        # 3. Assign points to the closest centroid
        cluster_assignments = torch.argmin(distances, dim=1)
        
        # 4. Update centroids by computing the mean of assigned points
        new_centroids = torch.zeros_like(centroids)
        for c in range(num_clusters):
            # Mask points matching current cluster
            assigned_points = X[cluster_assignments == c]
            if len(assigned_points) > 0:
                new_centroids[c] = assigned_points.mean(dim=0)
            else:
                # Handle empty cluster by re-initializing randomly
                new_centroids[c] = X[torch.randint(0, num_samples, (1,))]
        
        # Check for convergence (if centroids stop changing)
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
            
        centroids = new_centroids
        
    return cluster_assignments, centroids