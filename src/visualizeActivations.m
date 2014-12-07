function visualizeActivations(file, outputName)
% file is defined to be a MAT file with the following variable:
% numLayers - number of matrices stored in file
% activations - cell array of matrices (one matrix per layer), where each
%           matrix is (numFilters x filterDim x filterDim),
%           where the first matrix has the largest filterDim.
% A cell array is used because we have to store matrices of different
%           dimensions inside one structure.
% outputName is the prefix of the image files to be output.
borderWidth = 2;
magnifyRatio = 1; % Usually we need to upscale the output to be visible.

data = load(file, 'numLayers', 'activations');
numLayers = data.numLayers;
activations = data.activations;
assert(numLayers > 0, 'No layers to process!');
visualizeActivationsNoMAT(numLayers, activations, borderWidth, magnifyRatio, outputName);
end

function visualizeActivationsNoMAT(numLayers, activations, borderWidth, magnifyRatio, outputName)
% For each layer...
for i=1:numLayers
    layer = activations{i};
    % Resize the image for better visibility.
    layerResized = zeros(size(layer, 1), magnifyRatio * size(layer, 2), ...
        magnifyRatio * size(layer, 3));
    for j=1:size(layer, 1)
        square = squeeze(layer(j, :, :));
        resized = imresize(square, magnifyRatio);
        layerResized(j, :, :) = resized;
    end
    layer = layerResized;

    % Keep track of other paramaters.
    numFilters = size(layer, 1);
    filterDim = size(layer, 2);
    border = zeros(filterDim, borderWidth, 3); % 3 for RGB
    for j=1:3
        border(:, :, j) = ones(filterDim, borderWidth);
    end

    % For each filter...
    % add to our final image plus a border.
    newIm = [];
    for j=1:numFilters
        % Make this image greyscale in RGB representation
        newIm = [newIm repmat(squeeze(layer(j, :, :)), [1 1 3])];
        if j ~= numFilters
            newIm = [newIm border];
        end
    end

    % Save the image to PNG.
    imwrite(newIm, strcat(outputName, '_layer_', num2str(i), '.png'));
end
end