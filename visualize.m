function visualize(file, outputName)
% file is defined to be a MAT file with the following variable:
% numLayers - number of matrices stored in file
% filters - cell array of matrices (one matrix per layer), where each
%           matrix is (numFilters x filterDim x filterDim),
%   where the first matrix has the largest filterDim. A cell array is used
%   because we have to store matrices of different
%   dimensions inside one structure.
% outputName is the prefix of the image files to be output.
borderWidth = 16;
magnifyRatio = 15; % Usually we need to upscale the output to be visible.

data = load(file, 'numLayers', 'filters');
numLayers = data.numLayers;
filters = data.filters;
assert(numLayers > 0, 'No layers to process!');
visualizeNoMAT(numLayers, filters, borderWidth, outputName);
end

function visualizeNoMAT(numLayers, filters, borderWidth, magnifyRatio, outputName)
% For each layer...
for i=1:numLayers
    layer = filters{i};
    numFilters = size(layer, 1);
    filterDim = size(layer, 2);
    border = zeros(filterDim, borderWidth);
    
    % For each filter...
    % add to our final image plus a border.
    newIm = [];
    for j=1:numFilters
        newIm = [newIm layer(:, :, j)];
        if j ~= numFilters
            newIm = [newIm border];
        end
    end
    
    % Resize the image for better visibility.
    newIm = imresize(newIm, magnifyRatio);
    % Save the image to JPG.
    imwrite(newIm, strcat(outputName, '_layer_', num2str(i), '.png'));
end
end