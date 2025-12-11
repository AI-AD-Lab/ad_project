#%%
import pandas as pd
from pathlib import Path

from torch.utils.data import DataLoader


from torch.utils.data import Subset
from _utils.plot_utils import plot_2d_graph
from _utils.data_processing_utils import TrajectoryDataset, split_train_test_by_class




if __name__ == "__main__":
    # CONFIG
    GRANDPARENTS_DIR = Path(__file__).resolve().parent.parent
    SYN_LOG_DATA_ROOT_DIR = GRANDPARENTS_DIR / 'simulation_TOTAL_250626_2'

    dataset = TrajectoryDataset(SYN_LOG_DATA_ROOT_DIR)

    total_len = []

    train_idx, test_idx = split_train_test_by_class(
    dataset.df,
    label_col="trajectory type",
    test_ratio=0.3,
    seed=42,
    )

    train_dataset = Subset(dataset, train_idx)
    test_dataset  = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

    for batch_data, batch_labels, batch_file_names in train_loader:

        total_len.append(batch_data.shape[0])

        save_dir = Path("./output/shrink_data_2500/")
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        for i in set(batch_labels.numpy()):
            class_dir = save_dir / f"class_{i}"
            if not class_dir.exists():
                class_dir.mkdir(parents=True)

        for data, label, file_name in zip(batch_data, batch_labels, batch_file_names):
            data = pd.DataFrame(data, columns=['PositionX (m)','PositionY (m)','PositionZ (m)',
                                              'VelocityX(EntityCoord) (km/h)','VelocityY(EntityCoord) (km/h)','VelocityZ(EntityCoord) (km/h)',
                                              'AccelerationX(EntityCoord) (m/s2)', 'AccelerationY(EntityCoord) (m/s2)', 'AccelerationZ(EntityCoord) (m/s2)'])

            save_path = save_dir / f"class_{label.item()}" / f"{file_name.replace('.csv', '_shrink.csv')}"
            # data.to_csv(save_path, index=False)

            data['time (sec)'] = data.index * 0.2
            save_plot_path = save_dir / f"class_{label.item()}" / f"{file_name.replace('.csv', '.png')}"
            plot_2d_graph(data, x='time (sec)', y='AccelerationY(EntityCoord) (m/s2)', legend=False, save_path=str(save_plot_path))



# %%
