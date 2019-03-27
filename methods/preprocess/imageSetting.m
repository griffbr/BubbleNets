function [] = imageSetting(dataPath, dataName)
% Ex)
% dataPath = './data/rawData';
% dataName = 'dog-agility';

imgPath = [dataPath '/' dataName '/src'];
% ppmPath = [dataPath '/' dataName '/ppm'];

temp = dir(imgPath);
temp = temp(~ismember({temp.name},{'.','..','.DS_Store'}));
if ~exist(imgPath, 'dir') || isempty(temp)
    fprintf('\nBuilding src image folder for %s video.\n', dataName);
    if ~exist(imgPath, 'dir')
        mkdir(imgPath);
    end
    vidPath = [dataPath '/' dataName];
    vidList = dir([vidPath '/*.avi']);
    vidList = [vidList; dir([vidPath '/*.mj2'])];
    vidList = [vidList; dir([vidPath '/*.wmv'])];
    vidList = [vidList; dir([vidPath '/*.mov'])];
    vidList = [vidList; dir([vidPath '/*.mp4'])];
    
    vidList = [vidList; dir([vidPath '/*.3gp'])];
    vidList = [vidList; dir([vidPath '/*.mxf'])];
    vidList = [vidList; dir([vidPath '/*.asf'])];
    vidList = [vidList; dir([vidPath '/*.wav'])];
    vidList = [vidList; dir([vidPath '/*.mts'])];
    
    vidObj = VideoReader([vidPath '/' vidList(1).name]);
    k = 0;
    while hasFrame(vidObj)
        imwrite(readFrame(vidObj), sprintf('%s/%05i.png', imgPath, k));
        fprintf('%d.',k)
        k = k+1;
    end
end

