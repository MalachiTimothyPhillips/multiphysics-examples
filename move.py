import os

for root, dirs, files in os.walk('.'):
    for file in files:
        if not "tt_" in file:
            continue

        # split tt_(.+) -> multiphys_(.+)
        newFile = "multiphys_" + file.strip("tt_")
        os.rename(file, newFile)

