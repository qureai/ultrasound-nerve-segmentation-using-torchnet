--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Machine enclosing the torchnet engine
--]]

require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'utils/utils.lua'
tnt = require 'torchnet'

local Machine = torch.class 'Machine'

--- Class that sets engine, criterion, model
-- @param opt
function Machine:__init(opt)
   opt = opt or {}

   self.trainDataset = opt.trainDataset -- training dataset to be used
   self.valDataset = opt.valDataset -- validation dataset to be used
   self.trainSize = opt.trainSize or self.trainDataset:size() -- size of training dataset to be used
   self.valSize = opt.valSize or self.valDataset:size() -- size of validation dataset to be used

   self.trainBatchSize = opt.trainBatchSize or 32
   self.valBatchSize = opt.valBatchSize or 32

   self.model,self.modelName = self:LoadModel(opt) -- model to be used
   self.criterion = self:LoadCriterion(opt) -- criterion to be used
   self.engine = self:LoadEngine(opt) -- engine to be used

   self.savePath = opt.savePath -- path where models has to be saved
   self.maxepoch = opt.maxepoch -- maximum number of epochs for training
   self.dataset = opt.dataset -- name of the base file used for training
   self.learningalgo = opt.optimMethod -- name of the learning algorithm used

   self.meters = self:LoadMeters(opt) -- table of meters, key is the name of meter and value is the meter
   self:attachHooks(opt)
   self:setupEngine(opt)
end

--- Loads the model
-- @return Model loaded in CUDA,Name of the model
function Machine:LoadModel(opt)
   local model = opt.model
   require(model)
   local net,name = createModel(opt)
   net = net:cuda()
   cudnn.convert(net, cudnn)
   return net,name
end

--- Loads the criterion
-- @return Criterion loaded in CUDA
function Machine:LoadCriterion(opt)
   local weights = torch.Tensor(2)
   -- based on the ratio of distribution of masks pixels w.r.t no mask pixels
   weights[1] = 1/0.985
   weights[2] = 1/0.015
   local criterion = cudnn.SpatialCrossEntropyCriterion(weights)
   criterion = criterion:cuda()
   return criterion
end

--- Loads the engine
-- @return Optim Engine Instance
function Machine:LoadEngine(opt)
   local engine = tnt.OptimEngine()
   return engine
end

--- Loads all the meters
-- @return Table of meters such that, key is a string denoting meter name and value is the meter
-- Keys - Training Loss, Training Dice Score, Validation, Validation Dice Score, Param Norm, GradParam Norm, Norm Ratio, Time
function Machine:LoadMeters(opt)
   local meters = {}
   meters['Training Loss'] = tnt.AverageValueMeter()
   meters['Training Dice Score'] = tnt.AverageValueMeter()
   meters['Validation Loss'] = tnt.AverageValueMeter()
   meters['Validation Dice Score'] = tnt.AverageValueMeter()
   meters['Param Norm'] = tnt.AverageValueMeter()
   meters['GradParam Norm'] = tnt.AverageValueMeter()
   meters['Norm Ratio'] = tnt.AverageValueMeter()
   meters['Time'] = tnt.TimeMeter()
   return meters
end

--- Resets all the meters
function Machine:ResetMeters()
   for i,v in pairs(self.meters) do
      v:reset()
   end
end

--- Prints the values in all the meters
function Machine:PrintMeters()
   for i,v in pairs(self.meters) do
      io.write(('%s : %.5f | '):format(i,v:value()))
   end
end

--- Trains the model
function Machine:train(opt)
   self.engine:train{
      network   = self.model,
      iterator  = getIterator('train',self.trainDataset,self.trainBatchSize),
      criterion = self.criterion,
      optimMethod = self.optimMethod,
      config = self.optimConfig,
      maxepoch = self.maxepoch
   }
end

--- Test the model against validation data
function Machine:test(opt)
   self.engine:test{
      network   = self.model,
      iterator  = getIterator('test',self.valDataset,self.valBatchSize),
      criterion = self.criterion,
   }
end

--- Given the state, it will save the model as ModelName_DatasetName_LearningAlgorithm_epoch_torchnet_EpochNum.t7
function Machine:saveModels(state)
   local savePath = paths.concat(self.savePath,('%s_%s_%s_epoch_torchnet_%d.t7'):format(self.modelName,self.dataset,self.learningalgo,state.epoch))
   torch.save(savePath,state.network:clearState())
end

--- Adds hooks to the engine
-- state is a table of network, criterion, iterator, maxEpoch, optimMethod, sample (table of input and target),
-- config, optim, epoch (number of epochs done so far), t (number of samples seen so far), training (boolean denoting engine is in training or not)
-- https://github.com/torchnet/torchnet/blob/master/engine/optimengine.lua for position of hooks as to when they are called
function Machine:attachHooks(opt)

   --- Gets the size of the dataset or number of iterations
   local onStartHook = function(state)
      state.numbatches = state.iterator:execSingle('size')  -- for ParallelDatasetIterator
   end

   --- Resets all the meters
   local onStartEpochHook = function(state)
      if self.learningalgo == 'sgd' then
         state.optim.learningRate = self:LearningRateScheduler(state,state.epoch+1)
      end
      print(("Epoch : %d, Learning Rate : %.5f "):format(state.epoch+1,state.optim.learningRate or state.config.learningRate))
      self:ResetMeters()
   end

   --- Transfers input and target to cuda
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   local onSampleHook = function(state)
      igpu:resize(state.sample.input:size()):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end

   local onForwardHook = function(state)
   end

   --- Updates losses and dice score
   local onForwardCriterionHook = function(state)
      if state.training then
         self.meters['Training Loss']:add(state.criterion.output/state.sample.input:size(1))
         self.meters['Training Dice Score']:add(CalculateDiceScore(state.network.output,state.sample.target))
      else
         self.meters['Validation Loss']:add(state.criterion.output/state.sample.input:size(1))
         self.meters['Validation Dice Score']:add(CalculateDiceScore(state.network.output,state.sample.target))
      end
   end

   local onBackwardCriterionHook = function(state)
   end

   local onBackwardHook = function(state)
   end

   --- Update the parameter norm, gradient parameter norm, norm ratio and update progress bar to denote number of batches done
   local onUpdateHook = function(state)
      self.meters['Param Norm']:add(state.params:norm())
      self.meters['GradParam Norm']:add(state.gradParams:norm())
      self.meters['Norm Ratio']:add((state.optim.learningRate or state.config.learningRate)*state.gradParams:norm()/state.params:norm())
      xlua.progress(state.t,state.numbatches)
   end

   --- Sets t to 0, does validation and prints results of the epoch
   local onEndEpochHook = function(state)
      state.t = 0
      self:test()
      self:PrintMeters()
      self:saveModels(state)
   end

   local onEndHook = function(state)
   end

   --- Attaching all the hooks
   self.engine.hooks.onStart = onStartHook
   self.engine.hooks.onStartEpoch = onStartEpochHook
   self.engine.hooks.onSample = onSampleHook
   self.engine.hooks.onForward = onForwardHook
   self.engine.hooks.onForwardCriterion = onForwardCriterionHook
   self.engine.hooks.onBackwardCriterion = onBackwardCriterionHook
   self.engine.hooks.onBackward = onBackwardHook
   self.engine.hooks.onUpdate = onUpdateHook
   self.engine.hooks.onEndEpoch = onEndEpochHook
   self.engine.hooks.onEnd = onEndHook
end

--- Returns the learning for the epoch
-- @param state State of the training
-- @param epoch Current epoch number
-- @return Learning Rate
-- Training scheduler that reduces learning by factor of 10 rate after every 40 epochs
function Machine:LearningRateScheduler(state,epoch)
    local decay = 0
    local step = 1
    decay = math.ceil((epoch - 1) / 40)
    return math.pow(0.1, decay)
end

--- Sets up the optim engine based on parameter received
-- @param opt It must contain optimMethod
function Machine:setupEngine(opt)
   if opt.optimMethod=='sgd' then
      self.optimMethod = optim.sgd
      self.optimConfig = {
         learningRate = 0.1,
         momentum = 0.9,
         nesterov = true,
         weightDecay = 0.0001,
         dampening = 0.0,
      }
   elseif opt.optimMethod=='adam' then
      self.optimMethod = optim.adam
      self.optimConfig = {
         learningRate = 0.1
      }
   end
end

--- Iterator for moving over data
-- @param mode Either 'train' or 'test', defines whether iterator for training or testing
-- @param ds Dataset for the iterator
-- @param size Size of data to be used
-- @param batchSize Batch Size to be used
-- @return parallel dataset iterator
function getIterator(mode,ds,batchSize)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      transform = GetTransforms(mode),
      init = function()
         tnt = require 'torchnet'
      end,
      closure = function()
         return tnt.BatchDataset{
            batchsize = batchSize,
            dataset = ds
         }
      end
   }
end
