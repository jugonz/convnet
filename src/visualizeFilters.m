function visualizeFilters(file, outputName)
% file is defined to be a MAT file with the following variable:
% numLayers - number of matrices stored in file
% filters - cell array of matrices (one matrix per layer), where each
%           matrix is (numFilters x filterDim x filterDim),
%   where the first matrix has the largest filterDim. A cell array is used
%   because we have to store matrices of different
%   dimensions inside one structure.
% outputName is the prefix of the image files to be output.
borderWidth = 2;
magnifyRatio = 1; % Usually we need to upscale the output to be visible.
gain = 25/255; % How much to increase pixel values by.

data = load(file, 'numLayers', 'filters');
numLayers = data.numLayers;
filters = data.filters;
assert(numLayers > 0, 'No layers to process!');
visualizeFiltersNoMAT(numLayers, filters, borderWidth, magnifyRatio, gain, outputName);
end

function visualizeFiltersNoMAT(numLayers, filters, borderWidth, magnifyRatio, gain, outputName)
% For each layer...
for i=1:numLayers
    layer = filters{i};

    % Add the gain to each value in the image, clamping if needed.
    for j=1:size(layer, 1)
        for k=1:size(layer, 2)
            for l=1:size(layer, 3)
                gained = layer(j, k, l) + gain;
                if gained >= 1
                    gained = 1.;
                end
                layer(j, k, l) = gained;
            end
        end
    end

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
    border(:, :, 3) = ones(filterDim, borderWidth);

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