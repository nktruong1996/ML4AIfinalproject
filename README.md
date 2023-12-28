This project serves as a quick survey to some common dimensional reduction techniques. \
I tried to go over PCA, MDS, t-SNE and UMAP with: the theory behind them (not in-depth), advantages and disadvantages, comparing their results on basic datasets (digits, MNIST, Fashion MNIST) \
The section on exploratory data analysis with t-SNE and UMAP only contains the codes necessary for file preparations, the file addresses used in each .py file is personal to myself and should be editted accordingly. (this might change later) \
The file prepared above will then be fed to TensorBoard locally to generate a graph with the picture data as labels, which is highly beneficial for data analysis, however as TensorBoard only takes at most 5000 datapoints, improvements may be needed for large datasets. \
The directories containing the files for TensorBoard should be MNIST and FMNIST in this github page. \
Inside the the MNIST and FMNIST folders are:
* .zip files containing the png files, which are the picture data used in each respective dataset
* content folder containing all the png files
* .py files containing generative codes to configure the TensorBoard Projector function
* prompts.txt containing the command lines for running the TensorBoard Projector, generating the sprite image and the local link for TensorBoard
* sprite.png is the sprite image for the dataset used in the folder
* .tsv files are used to feed information to TensorBoard
* checkpoint, embedding and .pbtxt files are generated from the .py files to configure TensorBoard
