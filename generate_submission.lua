--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Generates submission file
--]]

require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'paths'
require 'image'
require 'hdf5'
require 'xlua'
require 'nngraph'
require 'csvigo'
require 'utils/utils.lua'
require 'constants.lua'

torch.setnumthreads(1) -- Increase speed
torch.setdefaulttensortype('torch.FloatTensor')

-- command line instructions reading
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Submission file generation script')
cmd:text()
cmd:text('Options:')
cmd:option('-dataset','data/test.h5','Testing dataset to be used')
cmd:option('-model','models/unet.t7','Path of the trained model to be used')
cmd:option('-csv','submisson.csv','Path of the csv file to be generated')
cmd:option('-testSize',5508,'Number of images for which data is to be generated - 5508 if all images on test set, 5635 if it is on train set')

-- Returns table of transformations
function GetTransformations()
	-- flag to check if transformations to be reapplied on label, set true for segmentation
	local transformations = {}
	for i=1,4 do
		transformations[i] = {}
	end
	transformations[1]['do'],transformations[1]['undo'] = HorizontalFlip()
	transformations[2]['do'],transformations[2]['undo'] = VerticalFlip()
	transformations[3]['do'],transformations[3]['undo'] = Rotation(1)
	transformations[4]['do'],transformations[4]['undo'] = Rotation(-1)
	return transformations
end

--- Returns generated masks given the model, dataset, baseProbability and testSize
-- @param opt A table that contains path for the model, dataset and testSize
function GenerateMasks(opt)
	print("Loading model and dataset")
	local model = torch.load(opt.model)
	model = model:cuda()
	model:evaluate()
	local xf = hdf5.open(opt.dataset)
	local testImages = xf:read('/images'):all()
	xf:close()
	local masks = torch.zeros(opt.testSize,trueHeight*trueWidth)
	print("Generating masks")
	local maskCount = 0
	for i=1,opt.testSize do
		-- scale the image and divide the pixel by 255
		local input = image.scale(testImages[i][1], imgWidth, imgHeight, interpolation)
		local modelOutput = GetSegmentationModelOutputs(model,input)
		masks[i] = modelOutput:t():reshape(trueWidth*trueHeight) -- taking transpose and reshaping it for being able to convert to RLE
		if GetLabel(masks[i]) == 2 then
			maskCount = maskCount + 1
		end
		xlua.progress(i,opt.testSize)
	end
	print(("Number of images with masks : %d"):format(maskCount))
	return masks
end

--- Returns the mask after taking average over augmentation of images
-- @param model Model to be used, loaded in CUDA
-- @param img Image to be used
function GetSegmentationModelOutputs(model,img)
	local transformations = GetTransformations()
	local modelOutputs = torch.Tensor(#transformations+1,trueHeight,trueWidth)
	modelOutputs[1] = GetMaskFromOutput(model:forward(img:reshape(1,1,imgHeight,imgWidth):cuda())[1],true)
	for i=1,#transformations do
		modelOutputs[i+1] = GetMaskFromOutput(model:forward(transformations[i]['do'](img):reshape(1,1,imgHeight,imgWidth):cuda())[1],true,transformations[i]['undo'])
	end
	return GetTunedResult(torch.mean(modelOutputs,1)[1],0.5)
end

--- Generates CSV given the masks and opt table containing csv path
function GenerateCSV(opt,masks)
	print("Generating RLE")
	-- rle encoding saved here, later written to csv
	local rle_encodings = {}
	rle_encodings[1] = {"img","pixels"}
	for i=1,opt.testSize do
		rle_encodings[i+1]={tostring(i),getRle(masks[i])}
		xlua.progress(i,opt.testSize)
	end
	-- saving the csv file
	csvigo.save{path=opt.csv,data=rle_encodings}
end

--- The main function that directs how movie is made
function GenerateSubmission(opt)
	local masks = GenerateMasks(opt)
	GenerateCSV(opt,masks)
end

local opt = cmd:parse(arg or {}) -- Table containing all the above options
GenerateSubmission(opt)
