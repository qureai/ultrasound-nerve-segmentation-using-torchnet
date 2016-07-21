--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Creates the dataset for use in HDF5 format
--]]

require 'paths'
require 'hdf5'
require 'xlua'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')
-- command line instructions reading
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 context encoder training script')
cmd:text()
cmd:text('Options:')
cmd:option('-train','','Path to train data')
cmd:option('-trainOutput','','Path to output train file to be generated')
cmd:option('-test','','Path to the test data')
cmd:option('-testOutput','','Path to test file to be generated')

function findImages(dir,ext)
    -- find options
    local findOptions = ' -iname "*.' .. ext .. '"'
    local f = io.popen('find -L ' .. dir .. findOptions)

    local maxLength = -1
    local imagePaths = {}

    -- Generate a list of all the images
    while true do
        local line = f:read('*line')
        if not line then break end

        local filename = paths.basename(line)
        local path = paths.dirname(line).. '/' .. filename
        table.insert(imagePaths, path)
    end

    f:close()
    return imagePaths
end

-- creates the h5 file given dir, ext, h5path,dsetName
function create_h5(dir,ext,h5path,dsetName)
    print("Creating test dataset")
    local pathsImages = findImages(dir,ext)
    local mTensor = torch.FloatTensor(#pathsImages,1,420,580)
    for i,v in ipairs(pathsImages) do
        xlua.progress(i,#pathsImages)
        local path = v
        local img = image.loadPNG(path)

        local imageNumber = tonumber(string.gsub(paths.basename(path),"."..ext,""),10)
        mTensor[imageNumber][1] = img
    end

    local myf = hdf5.open(h5path, 'w')
    myf:write(dsetName, mTensor)
    myf:close()
end

-- Find isn't returning in alphabetical order, reading images based on mask paths
function create_train_h5(dir, ext, h5path)
    print("Creating train dataset")
    local images = findImages(dir,ext)

    -- creating a count of list of number of images per patient
    local imageCounts = {}
    for i,v in ipairs(images) do
        -- xlua.progress(i,#images)
        local imagePath = v
        local patientNumber,_ = tonumber(string.gsub(paths.basename(v),"_%d+%."..ext,""),10)
        if imageCounts[patientNumber] then
            imageCounts[patientNumber] = imageCounts[patientNumber] + 1
        else
            imageCounts[patientNumber] = 1
        end
    end

    -- loading images and creating h5 files
    local myf = hdf5.open(h5path, 'w')

    for patientNumber,imageNumber in ipairs(imageCounts) do
        local imgTensor = torch.FloatTensor(imageNumber,1,420,580)
        local maskTensor = torch.FloatTensor(imageNumber, 420,580)
        for num=1,imageNumber do
            -- load image
            local imagePath = paths.concat(dir,patientNumber.."_"..num.."."..ext)
            imgTensor[num][1] = image.loadPNG(imagePath)

            -- load mask
            local maskPath = string.gsub(imagePath,"images","masks"):gsub('.'..ext,'_mask.'..ext)
            maskTensor[num] = image.loadPNG(maskPath)[1]
        end
        myf:write('/images_'..patientNumber,imgTensor)
        myf:write('/masks_'..patientNumber,maskTensor)
        xlua.progress(patientNumber,#imageCounts)
    end
    myf:close()
end

local opt = cmd:parse(arg or {}) -- Table containing all the above options
for i,v in pairs(opt) do
    if v == "" then
        opt[i] = nil
    end
end

-- for train images
-- Masks must be in /path/to/trainData/masks
if opt.train and opt.trainOutput then
    create_train_h5(opt.train,'png',opt.trainOutput)
end

-- for test images
if opt.test and opt.testOutput then
    create_h5(opt.test,'png',opt.testOutput,'/images')
end
