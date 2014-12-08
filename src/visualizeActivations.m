function visualizeActivations(file, outputName)
% file is defined to be a MAT file with the following variable:
% numLayers - number of matrices stored in file
% activations - cell array of matrices (one matrix per layer), where each
%           matrix is (numFilters x filterDim x filterDim),
%           where the first matrix has the largest filterDim.
% A cell array is used because we have to store matrices of different
%           dimensions inside one structure.
% outputName is the prefix of the image files to be output.
borderWidth = 1;
magnifyRatio = 1; % Usually we need to upscale the output to be visible.
gain = 0.5;

data = load(file, 'numLayers', 'activations');
numLayers = data.numLayers;
activations = data.activations;
assert(numLayers > 0, 'No layers to process!');
visualizeActivationsNoMAT(numLayers, activations, borderWidth, magnifyRatio, gain, outputName);
end

function visualizeActivationsNoMAT(numLayers, activations, borderWidth, magnifyRatio, gain, outputName)
% For each layer...
for i=1:numLayers
    layer = activations{i};
    % Add the gain to each value in the image, clamping if needed.
    for j=1:size(layer, 1)
        for k=1:size(layer, 2)
            for l=1:size(layer, 3)
                gained = gain*layer(j, k, l) + 0.5;
%                 if gained >= 1
%                     gained = 1.;
%                 end
                layer(j, k, l) = gained;
            end
        end
    end

    % Resize the image for better visibility.
%     layerResized = zeros(size(layer, 1), magnifyRatio * size(layer, 2), ...
%         magnifyRatio * size(layer, 3));
%     for j=1:size(layer, 1)
%         square = squeeze(layer(j, :, :));
%         resized = imresize(square, magnifyRatio);
%         layerResized(j, :, :) = resized;
%     end
%     layer = layerResized;

    % Keep track of other paramaters.
    numFilters = size(layer, 1);
    filterDim = size(layer, 2);
    border = zeros(filterDim, borderWidth, 3); % 3 for RGB
    border(:, :, 3) = ones(filterDim, borderWidth);
    
    %%%%
    numPerRow = 10;
    gridIm = ones(((numFilters/numPerRow)*filterDim) + ((numFilters/numPerRow)+1)*borderWidth, (numPerRow*filterDim) + (numPerRow+1)*borderWidth);
    
    start_i = 1 + borderWidth;
    end_i = filterDim + borderWidth;
    start_j = 1 + borderWidth;
    end_j = filterDim + borderWidth;
    for k=1:numFilters
        gridIm(start_i:end_i,start_j:end_j) = squeeze(layer(k,:,:));
        
        start_j = start_j + filterDim + borderWidth;
        end_j = end_j + filterDim + borderWidth;
        
        if mod(k,numPerRow) == 0
            start_i = start_i + filterDim + borderWidth;
            end_i = end_i + filterDim + borderWidth;
            
            start_j = 1 + borderWidth;
            end_j = filterDim + borderWidth;
        end
    end
    %%%%

    % For each filter...
    % add to our final image plus a border.
%     newIm = [];
%     for j=1:numFilters
%         % Make this image greyscale in RGB representation
%         newIm = [newIm repmat(squeeze(layer(j, :, :)), [1 1 3])];
%         if j ~= numFilters
%             newIm = [newIm border];
%         end
%     end

    % Save the image to PNG.
    imwrite(gridIm, strcat(outputName, '_layer_', num2str(i), '.png'));
end
% For each layer...
% for i=1:numLayers
%     layer = activations{i};
%     % Resize the image for better visibility.
%     layerResized = zeros(size(layer, 1), magnifyRatio * size(layer, 2), ...
%         magnifyRatio * size(layer, 3));
%     for j=1:size(layer, 1)
%         square = squeeze(layer(j, :, :));
%         resized = imresize(square, magnifyRatio);
%         layerResized(j, :, :) = resized;
%     end
%     layer = layerResized;
%     
%     % Keep track of other paramaters.
%     numFilters = size(layer, 1);
%     filterDim = size(layer, 2);
%     border = zeros(filterDim, borderWidth, 3); % 3 for RGB
%     border(:, :, 3) = ones(filterDim, borderWidth);
%     
%     % For each filter...
%     % add to our final image plus a border.
%     newIm = [];
%     for j=1:numFilters
%         % Make this image greyscale in RGB representation
%         newIm = [newIm repmat(squeeze(layer(j, :, :)), [1 1 3])];
%         if j ~= numFilters
%             newIm = [newIm border];
%         end
%     end
%     
%     % Save the image to PNG.
%     imwrite(newIm, strcat(outputName, '_layer_', num2str(i), '.png'));
% end
end