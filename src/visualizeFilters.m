function visualizeFilters(file, outputName)
% file is defined to be a MAT file with the following variable:
% numLayers - number of matrices stored in file
% filters - cell array of matrices (one matrix per layer), where each
%           matrix is (numFilters x filterDim x filterDim),
%   where the first matrix has the largest filterDim. A cell array is used
%   because we have to store matrices of different
%   dimensions inside one structure.
% outputName is the prefix of the image files to be output.
borderWidth = 1;
magnifyRatio = 1; % Usually we need to upscale the output to be visible.
gain = 25; % Multiplicative increase for pixel values.

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
    % Shift values over to be > 0, and multiply by our gain.
    for j=1:size(layer, 1)
        for k=1:size(layer, 2)
            for l=1:size(layer, 3)
                gained = gain*layer(j, k, l) + 0.5;
                layer(j, k, l) = gained;
            end
        end
    end

    % Keep track of other paramaters.
    numFilters = size(layer, 1);
    filterDim = size(layer, 2);
    %border = ones(filterDim, borderWidth, 3); % 3 for RGB

    % Keep track of other paramaters.
    numPerRow = 20;
    gridIm = ones(((numFilters/numPerRow)*filterDim) + ((numFilters/numPerRow)+1)*borderWidth, (numPerRow*filterDim) + (numPerRow+1)*borderWidth);

    start_i = 1 + borderWidth;
    end_i = filterDim + borderWidth;
    start_j = 1 + borderWidth;
    end_j = filterDim + borderWidth;
    % Make a grid out of this image.
    for k=1:numFilters
        gridIm(start_i:end_i,start_j:end_j) = squeeze(layer(k,:,:));

        start_j = start_j + filterDim + borderWidth;
        end_j = end_j + filterDim + borderWidth;

        % End of row.
        if mod(k,numPerRow) == 0
            start_i = start_i + filterDim + borderWidth;
            end_i = end_i + filterDim + borderWidth;

            start_j = 1 + borderWidth;
            end_j = filterDim + borderWidth;
        end
    end

    % Save the image to PNG.
    imwrite(gridIm, strcat(outputName, '_layer_', num2str(i), '.png'));
end
end
