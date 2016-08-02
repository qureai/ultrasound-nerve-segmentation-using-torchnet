require 'image'
require 'nn'
require 'cudnn'
require 'cunn'
require 'constants.lua'
Map = require 'pl.Map'

--- Returns rle of a vector as a string
-- @param vec Must contain only 0s and 1s
function getRle(vec)
	local m = Map{}
	local is_one = false
	local num = 0
	local length = 0
	local n = vec:size(1)
	for i=1,n do
		if vec[i] == 1 then
			if is_one then
				length = length + 1
			else
				is_one = true
				length = 1
				num = i
			end
		else
			if is_one then
				is_one = false
				m:set(num,length)
			else
				num = 0
			end
		end
	end
	if is_one then
		is_one = false
		m:set(num,length)
	end
	-- concatenating pixels to form strings
	local rle_table = {}
	for i,v in ipairs(m:items()) do
		rle_table[#rle_table+1] = v[1]
		rle_table[#rle_table+1] = v[2]
	end
	return table.concat(rle_table,' ')
end

--- Returns class number based on mask, if mask exists class number is 2, else 1
function GetLabel(mask)
    if mask:sum() > 0 then
        return 2
    end
    return 1
end

--- For upscaling 112*144 image to 420*580 image using spatial nearest up sampling
function GetScaledUpImage(img)
	local scaleUpNN = nn.Sequential():add(nn.SpatialUpSamplingNearest(4)):add(nn.SpatialZeroPadding(2,2,-14,-14)):cuda()
	img = img:cuda()
	return scaleUpNN:forward(img)
end

--- Returns masks with pixel-wise probabilities
function GetMaskProbabilities(vec)
	local spatialSoftMax = nn.Sequential():add(cudnn.SpatialSoftMax()):cuda()
	vec = vec:cuda()
	return spatialSoftMax:forward(vec)
end

--- Returns do and undo function for horizontally flipping an image
function HorizontalFlip()
	return function(img) return image.hflip(img:float()) end, function(img) return image.hflip(img:float()) end
end

--- Returns do and undo function for vertically flipping an image
function VerticalFlip()
	return function(img) return image.vflip(img:float()) end, function(img) return image.vflip(img:float()) end
end

--- Returns do and undo function for rotating an image by specified degrees
-- @param deg Degrees with which to rotate
function Rotation(deg)
	local rot = deg * math.pi / 180
	return function(img) return image.rotate(img,rot,'bilinear') end, function(img) return image.rotate(img,-1*rot,'bilinear') end
end

--- Converts a table of tensors to tensor
function TableToTensor(table)
	local tensorSize = table[1]:size()
	local tensorSizeTable = {-1}
	for i=1,tensorSize:size(1) do
		tensorSizeTable[i+1] = tensorSize[i]
	end
	local merge=nn.Sequential():add(nn.JoinTable(1)):add(nn.View(unpack(tensorSizeTable)))
	return merge:forward(table)
end

function CalculateDiceScore(outputs, targets)
    -- Dice loss function calculator
    local dice_coeff = 0
    for i=1,outputs:size(1) do
        local output_flatten = GetMaskFromOutput(outputs[i])
        local target_flatten = targets[i]:float():add(-1)
        local numerator = torch.cmul(output_flatten, target_flatten)
        if output_flatten:sum() + target_flatten:sum() ~= 0 then
            dice_coeff = dice_coeff + 2*(numerator:sum())/(output_flatten:sum() + target_flatten:sum())
        else
            dice_coeff = dice_coeff + 1
        end
    end
    return dice_coeff/outputs:size(1)
end

function GetTunedResult(image, prob)
	return image:gt(prob):float()
end

--- Returns the mask given the output from unet, does resizing to original image if sizing set true
function GetMaskFromOutput(output,sizing,callback)
	local outputsoftmax = GetMaskProbabilities(output)
	if callback then
		outputsoftmax = callback(outputsoftmax:float())
	end
	if sizing then
		outputsoftmax = GetScaledUpImage(outputsoftmax)
	end
	return GetTunedResult(outputsoftmax[2],baseSegmentationProb):float()
end
