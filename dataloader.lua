--[[
This code is part of Ultrasound-Nerve-Segmentation Program

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Data Loader used to load nerve segmentation data
--]]


require 'hdf5'
dofile ('constants.lua')
local tnt = require 'torchnet'
local t = dofile ('utils/transforms.lua')

local DataLoader = torch.class 'DataLoader'
torch.setnumthreads(1)

--- Initializes Data Loader class by setting up train batch sizes and validation sizes
-- @param opt Takes options table with trainBatchSize and valBatchSize
function DataLoader:__init(opt)
    opt = opt or {}

    -- parameters
    self.trainBatchSize = opt.trainBatchSize or 8
    self.valBatchSize = opt.valBatchSize or 2
    self.testBatchSize = 1
    self.trainData = opt.trainData
    self.testData = opt.testData
    self.opt = opt

    self:Setup(opt)

end

--- Completes the setup of the data loaders
-- @param opt Must contain a cross validation parameter, and hence patients whose number is such that patientNum%5==cvParam, then that patients data is used for validation, else for training
function DataLoader:Setup(opt)
    print("Setting up data loader using ".. opt.dataset)
    local cvParam = opt.cvParam or -1

    -- load the complete data
    local xf = hdf5.open(opt.dataset)
    local fullData = xf:all() -- contains the complete data set
    xf:close()

    self.trainImages = {} -- contains the images used for train set
    self.trainMasks = {} -- contains the masks used for train set

    self.valImages = {} -- contains the images used for validation set
    self.valMasks = {} -- contains the masks used for validation set

    for i,v in pairs(fullData) do
        local i_string = tostring(i)
        if string.find(i_string,"images") then
            local patientNumber = tonumber(i_string:gsub("images_",""),10)
            local masks = fullData[i_string:gsub("images","masks")]
            for j=1,v:size(1) do
                if patientNumber%5 == cvParam then
                    self.valImages[#self.valImages+1] = v[j]
                    self.valMasks[#self.valMasks+1] = masks[j]:add(1)
                else
                    self.trainImages[#self.trainImages+1] = v[j]
                    self.trainMasks[#self.trainMasks+1] = masks[j]:add(1)
                end
            end
        end
    end
    print("Data loader setup done!")
end

--- Returns a list dataset
-- @param mode Defines what data must be returned, train or val
-- @param size Defines size of data needed
function DataLoader:GetData(mode,size)
    local images, masks
    if mode == 'train' then
        images = self.trainImages
        masks = self.trainMasks
    else
        images = self.valImages
        masks = self.valMasks
    end
    local dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1,#images):long(),
            load = function(idx)
                return { input = images[idx], target = masks[idx] }
            end,
        },
        size = size
    }
    return dataset
end

--- Returns the composition of transforms to be applied on dataset
-- @param mode Defines transformation for what data is needed
function GetTransforms(mode)
    if mode == 'train' then
        return GetTrainTransforms()
    else
        return GetValTransforms()
    end
end

--- Returns transform function used for training
function GetTrainTransforms()
    local f = function(sample)
        local images = sample.input
        local labels = sample.target
        local transforms = t.Compose{
            t.OneHotLabel(2),
            t.Resize(imgWidth, imgHeight),
            t.HorizontalFlip(0.5),
            t.Rotation(5),
            t.VerticalFlip(0.5),
            t.ElasticTransform(100,20),
            t.CatLabel()
        }
        local imagesTransformed = torch.Tensor(images:size(1),1,imgHeight,imgWidth)
        local masksTransformed = torch.Tensor(images:size(1),imgHeight,imgWidth)
        for i=1,images:size(1) do
            imagesTransformed[i],masksTransformed[i] = transforms(images[i]:float(),labels[i]:float())
        end
        sample['input'] = imagesTransformed
        sample['target'] = masksTransformed
        return sample
    end
    return f
end

--- Returns validation function used for training
function GetValTransforms()
    local f = function(sample)
        local images = sample.input
        local labels = sample.target
        local transforms = t.Compose{
            t.OneHotLabel(2),
            t.Resize(imgWidth, imgHeight),
            t.CatLabel()
        }
        local imagesTransformed = torch.Tensor(images:size(1),1,imgHeight,imgWidth)
        local masksTransformed = torch.Tensor(images:size(1),imgHeight,imgWidth)
        for i=1,images:size(1) do
            imagesTransformed[i],masksTransformed[i] = transforms(images[i]:float(),labels[i]:float())
        end
        sample['input'] = imagesTransformed
        sample['target'] = masksTransformed
        return sample
    end
    return f
end
