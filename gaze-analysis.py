from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#filter through data that is irrelevant (ie. gaze points down, head points down)
def csv_Filter (file):
    #list of frames that are relevant
    frames = []
    
    #read the file
    data = pd.read_csv(file)

    #iterate through each row and collect the frames that satisfy the filters
    for row in data.itertuples():
        if ((row[4] == 1) or ((row[6] > 0) and (row[9] > 0)) or (row[12] > 0)):
            frames.append(row[0])

    #collect the frames that satisfy the filter and return a dataframe with only those frames
    filtered = data.loc[frames, :]
    return filtered

def csv_plot (file):
    #get the path for where the plots are going to chill in
    csv_plots = Path("../Thesis/gaze_0_heatmaps")
    
    #read the file into a dataframe
    data = pd.read_csv(file)

    #read x and y coordinates for gaze angles
    #x = data.loc[:, "gaze_angle_x"]
    #y = data.loc[:, "gaze_angle_y"]

    #read x and y coordinated for gaze_0_x and gaze_0_y
    x = data.loc[:, "gaze_0_x"]
    y = data.loc[:, "gaze_0_y"]

    #generate heatmap
    heatmap, xAxis, yAxis = np.histogram2d(x, y, bins=100)
    extent = [xAxis[2], xAxis[-2], yAxis[2], yAxis[-2]]

    plt.clf()
    plt.imshow(heatmap, extent=extent, origin='lower')
    plt.savefig(csv_plots / file.stem, bbox_inches = 'tight')


    #TODO: figure out how to save the images into a separate folder under the csv_file names OR have all the datapoints in one image??



def main():
    """
    ### EXTRACTING RELATED COLUMNS ###
    #paths to the processed videos and the csv files with the constraints
    processed = Path("../Processed")
    csv_Files = Path("../Thesis/csvfiles")

    #sorted array of each file with a csv ending in the processed directory
    files = [ file for file in processed.iterdir() if file.suffix == ".csv" ]
    files.sort()

    #iterate through each file in the files array and extract the columns with the constraints into separate csv files
    for file in files:
        data = pd.read_csv(file)
        constraints = data.loc[:, ["frame", "timestamp", "confidence", "success", "gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry", "pose_Rz"]]
        #attempt to store the columns for each file in a separate csv in another directory
        constraints.to_csv(csv_Files / file.name, index=False)
    

    ### EXTRACTING SUCCESSFUL FRAMES ###
    #paths to the constrained files and to the destination directory
    csvFiles = Path("../Thesis/csvfiles")
    filtered_csv_files = Path("../Thesis/filtered_csv_files")

    #sorted array of constrained files
    files = [ file for file in csvFiles.iterdir() ]
    files.sort()

    #iterate through the constrained csv files, and filter the ones that are successful and looking up into separate csv files
    for file in files:
        filteredFile = csv_Filter(file)
        filteredFile.to_csv(filtered_csv_files / file.name, index=False)

    """
    ### GENERATING HEATMAPS ###
    #path to the csv files
    filtered_csv_files = Path("../Thesis/filtered_csv_files")
    csv_plots = Path("../Thesis/gaze_0_heatmaps")
    """
    #sorted array of csv files
    files = [ file for file in filtered_csv_files.iterdir() ]
    files.sort()
    
    #generate a heatmap for each csv file
    for file in files:
        csv_plot(file)
    
    #combine the csv files into one file to make the aggregated heatmap
    combined_csv = pd.concat([pd.read_csv(f) for f in files])
    combined_csv.to_csv(filtered_csv_files / 'combined_csv.csv', index=False)

    """
    #plot gaze_angle_x and gaze_angle_y
    data = pd.read_csv(filtered_csv_files / 'combined_csv.csv')
        
    #extract the x and y values
    x = data.loc[:, "gaze_0_x"]
    y = data.loc[:, "gaze_0_y"]

    #make the heatmap
    heatmap, xAxis, yAxis = np.histogram2d(x, y, bins=1000)
    extent = [xAxis[1], xAxis[-1], yAxis[1], yAxis[-1]]

    plt.clf()
    plt.imshow(heatmap, extent=extent, origin='lower')

    #grid parameters, superimposed on the heatmap
    ax = plt.gca()
    ax.set_xticks(np.arange(-1, 1, 1))
    ax.set_xticks(np.arange(-1, 1, 0.1), minor=True)
    ax.set_yticks(np.arange(-1, 1, 1))
    ax.set_yticks(np.arange(-1, 1, 0.1), minor=True)
    ax.grid(which = 'minor', alpha = 0.3)
    ax.grid(which = 'major', alpha = 0.7)

    #save the resulting heatmap
    plt.savefig(csv_plots / 'aggregated_plot_gaze_0.png', bbox_inches = 'tight', dpi=300)




    


if __name__ == "__main__":
    main()