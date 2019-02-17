%%   Segment White Blood Cell Nuclei function
%   _______________________________________________________________________
%   Le Duc Khai
%   Bachelor in Biomedical Engineering
%   FH Aachen - University of Applied Sciences, Germany.

%   Last updated on 12.02.2019.

%   The proposed algorithm segments automatically white blood cell nuclei.
%   This function works well with microscopic images, which have 3 types of
%   cells: red blood cells, white blood cells and platelets. Only white
%   blood cells possess nuclei.

%   Implementation is based on this scientific paper:
%       Mrs. Sonali C. Sonar, Prof. K. S. Bhagat
%       "An Efficient Technique for White Blood Cells Nuclei Automatic Segmentation"
  
%   The following codes are implemented only for PERSONAL USE, e.g improving
%   programming skills in the domain of Image Processing and Computer Vision.
%   If you use this algorithm, please cite the paper mentioned above to support
%   the authors.

%   Parameters:
%       image: the input image
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
function result_image = WBC_nuclei_segment(image)
%% Read the input image
original_image = imread(image);
if ndims(original_image) == 3
    image = rgb2gray(original_image);
else
    image = original_image;
end
image = uint8(image);
figure(1);
subplot(3,3,1); imshow(original_image); title('1. Original image'); 
subplot(3,3,2); imshow(image); title('2. Gray scale image'); 

%% Contrast enhancement
image_adjust = imadjust(image); % Adjust intensity values of the image
image_histo = histeq(image); % Use histogram equalization
% Addition to brighten most of details in image except the nuclei
image_add = imadd(image_adjust, image_histo); 
% Subtraction to highlight objects and borders including the nuclei
image_subtract = imsubtract(image_add, image_histo); 
% Addition to remove all other components but the nuclei
image = imadd(image_add, image_subtract); 
subplot(3,3,3); imshow(image_add); title('3. Addition image'); 
subplot(3,3,4); imshow(image_subtract); title('4. Subtraction image'); 
subplot(3,3,5); imshow(image); title('5. Added-again image'); 
% clear image_adjust; clear image_histo; clear image_add; clear image_subtract;

%% Minimum filter
B = zeros(size(image));
A = padarray(image,[1 1]); % Pad the image
x=[1:3]';
y=[1:3]';       
for i = 1 : size(A, 1) - 2
    for j = 1 : size(A, 2) - 2      
       % Vectorized method 
       window = reshape(A(i + x - 1, j + y - 1), [], 1);
       % Find the minimum of the selected window
       B(i,j) = min(window);
    end
end
B = uint8(B); % Convert to 0 - 255 intensity scale
image = B;
subplot(3,3,6); imshow(image); title('6. Minimum-filtered image'); 
clear i; clear j;
% clear A; clear B; clear x; clear y;  clear window;

%% Global image threshold
image = imbinarize(image, graythresh(image));
image = imcomplement(image);
subplot(3,3,7); imshow(image); title('7. Otsu-threshold image'); 
% clear counts; clear x;

%% Morphological opening
se = strel('disk', 9);
image = imopen(image, se); % Morphological opening to remove small high-pixel groups
subplot(3,3,8); imshow(image, []); title('8. Morphological opening image'); 

%% Connected components
image_cc = bwconncomp(image); % Find connected components
% Calculate number of pixels of each connected component
numPixels = cellfun(@numel, image_cc.PixelIdxList); 
numPixels_max = max(numPixels);
for i = 1 : size(numPixels, 2)
    if numPixels(i) < 0.5*numPixels_max
        % Remove components having less than 50% area of the biggest connected components
        % Change percentage of area if necessary
        image(image_cc.PixelIdxList{i}) = 0; 
    end
end
subplot(3,3,9); imshow(image); title('9. Connected components after size test'); 

%% Label image results
result_image = labeloverlay(original_image, image);
figure(2);
subplot(1,3,1); imshow(original_image); title('Original image'); 
subplot(1,3,2); imshow(result_image); title('Nuclei segmented image'); 
subplot(1,3,3); imshow(image); title('Nuclei shape');

end