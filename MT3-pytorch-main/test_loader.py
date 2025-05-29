from data.MAESTRO_loader import MIDIDatasetTab
from torch.utils.data import DataLoader

def test_MIDIDatasetTab():
    # Create an instance of MIDIDatasetTab
    dataset = MIDIDatasetTab(
        root_dir='/storage/nikolocal/mag/MT3-pytorch-main/data/datasets/maestro-v3.0.0',
        type='validation'
    )

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=True)

    # Iterate through the DataLoader and print the first batch
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("Inputs shape:", batch['inputs'].shape)
        print("Targets shape:", batch['targets'].shape)
        if 'end' in batch:
            print("End flag:", batch['end'])
        #break  # Stop after the first batch

if __name__ == "__main__":
    test_MIDIDatasetTab()