--[[
This code is part of Ultrasound-Nerve-Segmentation using Torchnet

Copyright (c) 2016, Qure.AI, Pvt. Ltd.
All rights reserved.

Initialization for the U-Net model
--]]

require 'nn'
require 'cudnn'
local nninit = require 'nninit'

local getBias = function(module)
	return module.bias
end

-- Kaiming initialization
local function MSRinit(net)

	local function init_kaiming(name)
		for k,v in pairs(net:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			v.bias:zero()
		end
	end
	init_kaiming('nn.SpatialConvolution')
	init_kaiming('nn.SpatialFullConvolution')

	local function init_bias(name)
		for k,v in pairs(net:findModules(name)) do
			v.bias:zero()
		end
	end
	init_bias('SpatialBatchNormalization')
end

return MSRinit
