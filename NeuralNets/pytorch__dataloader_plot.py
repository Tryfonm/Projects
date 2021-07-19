from matplotlib import pyplot as plt


class MyUtils():
    def __init__(self, dataloader):
        """

        Parameters
        ----------
        dataloader : Expects input of type torch.DataLoader
        """
        from matplotlib import pyplot as plt
        self.dl = dataloader
        self.dl_bs = dataloader.batch_size

    def visualize_batch(self, rows=4, cols=4, x_size=15, y_size=15, requested_batch=0):
        """
        Parameters
        ----------
        rows : Output rows to printout
        cols : Output columns to printout
        x_size : Horizontal dimension for subplot figure
        y_size : Vertical dimension for subplot figure
        requested_batch: Batch number to output, by default the first one is returned

        Returns
        -------
        None
        """
        if rows * cols <= self.dl_bs:
            fig, ax = plt.subplots(rows, cols, figsize=(x_size, y_size))
            for batch, (X, y) in enumerate(self.dl):
                if batch == requested_batch:
                    idx = 0
                    for i in range(rows):
                        for j in range(cols):
                            ax[i, j].imshow(X[idx].permute(1, 2, 0))
                            ax[i, j].axis('off')
                            idx += 1
        else:
            raise Warning('The requested number of images (rows x cols) is greater than the batch size.')
