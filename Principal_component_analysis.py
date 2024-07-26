import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import matplotlib.transforms as transforms

faces = fetch_lfw_people(min_faces_per_person=50)
# Who are these people?!
print(faces.target_names)
# What do their faces look like?
print(faces.images.shape)
# The target name index for each image (0 = Ariel Sharon, etc)
print(faces.target.shape)
print(faces.target)

fig, ax = plt.subplots(figsize=(4, 4.75))
ax.imshow(faces.images[12], cmap="binary_r")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
fig.tight_layout()

#######################  Use PCA to reduce the images  #######################

X = faces.data  # Image data
# print(X)
n_samples, n_features = X.shape

# Perform PCA
n_components = 150  # Number of components
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X)

# Visualize the first 50 eigenfaces
n_row = 5
n_col = 10
eigenfaces = pca.components_.reshape((n_components, faces.images.shape[1], faces.images.shape[2]))

plt.figure(figsize=(15, 8))
for i in range(n_row * n_col):
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title(f'Eigenface {i+1}')

plt.suptitle('Eigenfaces', fontsize=16)
plt.tight_layout()
plt.show()
plt.savefig('q5p1.png')

comp1 = np.linspace(0,len(X[0]),len(X[0]))
V = pca.components_
print(pca.components_)

plt.figure()
plt.scatter(pca.components_[1],pca.components_[0])
plt.quiver(V[0,0],V[0,1])
plt.quiver(V[1,0],V[1,1])
plt.axis('equal')
plt.show()


pca = PCA().fit(X)
print('cumsum', np.cumsum(pca.explained_variance_ratio_)[150])

# Plot cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()

weights = np.dot(X - pca.mean_, pca.components_[:n_components].T)
print(weights)
X_reconstructed = np.dot(weights, pca.components_[:n_components]) + pca.mean_

# Plotting original and reconstructed images
fig, axes = plt.subplots(nrows=len(faces.target_names), ncols=2, figsize=(10, 20))
for i, target_name in enumerate(faces.target_names):
    # Finding the index of the first image of each person
    index = np.where(faces.target == i)[0][0]
    
    # Original Image
    axes[i, 0].imshow(faces.images[index], cmap='gray')
    axes[i, 0].set_title(f'Original Image of {target_name}')
    axes[i, 0].axis('off')
    
    # Reconstructed Image
    reconstructed_image = X_reconstructed[index].reshape(faces.images.shape[1], faces.images.shape[2])
    axes[i, 1].imshow(reconstructed_image, cmap='gray')
    axes[i, 1].set_title(f'Reconstructed Image of {target_name}')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()
