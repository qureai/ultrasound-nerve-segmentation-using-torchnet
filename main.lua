--[[
This code is part of Ultrasound-Nerve-Segmentation Program

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Main file
--]]

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
tnt = require 'torchnet'

torch.setnumthreads(1) -- speed up
torch.setdefaulttensortype('torch.FloatTensor')

-- command line instructions reading
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 context encoder training script')
cmd:text()
cmd:text('Options:')
cmd:option('-dataset','data/train.h5','Training dataset to be used')
cmd:option('-model','models/unet.lua','Path of the model to be used')
cmd:option('-trainSize',100,'Size of the training dataset to be used, -1 if complete dataset has to be used')
cmd:option('-valSize',25,'Size of the validation dataset to be used, -1 if complete validation dataset has to be used')
cmd:option('-trainBatchSize',64,'Size of the batch to be used for training')
cmd:option('-valBatchSize',32,'Size of the batch to be used for validation')
cmd:option('-savePath','data/saved_models/','Path to save models')
cmd:option('-optimMethod','sgd','Algorithm to be used for learning - sgd | adam')
cmd:option('-maxepoch',250,'Epochs for training')
cmd:option('-cvParam',2,'Cross validation parameter used to segregate data based on patient number')

--- Main execution script
function main(opt)
   opt.trainSize = opt.trainSize==-1 and nil or opt.trainSize
   opt.valSize = opt.valSize==-1 and nil or opt.valSize

   -- loads the data loader
   require 'dataloader.lua'
   local dl = DataLoader(opt)
   local trainDataset = dl:GetData('train',opt.trainSize)
   local valDataset = dl:GetData('val',opt.valSize)
   opt.trainDataset = trainDataset
   opt.valDataset = valDataset
   opt.dataset = paths.basename(opt.dataset,'.h5')
   print(opt)

   require 'machine.lua'
   local m = Machine(opt)
   m:train()
end

local opt = cmd:parse(arg or {}) -- Table containing all the above options
main(opt)
