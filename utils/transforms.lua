--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--

require 'image'

local M = {}
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

function M.Compose(transforms)
    return function(input,label)
        for _, transform in ipairs(transforms) do
            input, label = transform(input, label)
        end
        return input, label
    end
end


function M.OneHotLabel(nclasses)
    return function(input, label)
        local oneHot = torch.Tensor(nclasses,label:size(1),label:size(2))
        for i = 1,nclasses do
            oneHot[i] = label:eq(i)
        end
        label = oneHot
        return input, label
    end
end

function M.CatLabel()
    return function(input, label)
        _, ar = torch.max(label, 1)
        label = ar[1]
        return input, label
    end
end


--- [[Structural Noise]] ---
function M.ElasticTransform(alpha, sigma)
    return function (input, label)
        H = input:size(2)
        W = input:size(3)
        filterSize = math.max(5,math.ceil(3*sigma))

        flow = torch.rand(2, H, W)*2 - 1
        kernel = image.gaussian(filterSize, sigma, 1, true)
        flow = image.convolve(flow, kernel, 'same')*alpha

        return image.warp(input, flow), image.warp(label, flow)
    end
end


function M.Scale(size, interpolation)
    interpolation = interpolation or 'bicubic'
    return function(input, label)
        local w, h = input:size(3), input:size(2)
        if (w <= h and w == size) or (h <= w and h == size) then
            return input, label
        end
        if w < h then
            return image.scale(input, size, h/w * size, interpolation), image.scale(label, size, h/w * size, interpolation)
        else
            return image.scale(input, w/h * size, size, interpolation), image.scale(label, w/h * size, size, interpolation)
        end
    end
end

function M.Resize(width, height, interpolation)
    interpolation = interpolation or 'bicubic'
    return function(input, label)
        return image.scale(input, width, height, interpolation), image.scale(label, width, height, interpolation)
    end
end

-- Crop to centered rectangle
function M.CenterCrop(size)
    return function(input, label)
        local w1 = math.ceil((input:size(3) - size)/2)
        local h1 = math.ceil((input:size(2) - size)/2)
        return image.crop(input, w1, h1, w1 + size, h1 + size), image.crop(label, w1, h1, w1 + size, h1 + size) -- center patch
    end
end

-- Random crop form larger image with optional zero padding
function M.RandomCrop(size, padding)
    padding = padding or 0

    return function(input, label)
        if padding > 0 then
            local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
            temp:zero()
            :narrow(2, padding+1, input:size(2))
            :narrow(3, padding+1, input:size(3))
            :copy(input)
            input = temp
        end

        local w, h = input:size(3), input:size(2)
        if w == size and h == size then
            return input
        end

        local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
        local out = image.crop(input, x1, y1, x1 + size, y1 + size)
        assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
        return out, image.crop(label, x1, y1, x1 + size, y1 + size)
    end
end

-- Resized with shorter side randomly sampled from [minSize, maxSize] (ResNet-style)
function M.RandomScale(minSize, maxSize)
    return function(input, label)
        local w, h = input:size(3), input:size(2)

        local targetSz = torch.random(minSize, maxSize)
        local targetW, targetH = targetSz, targetSz
        if w < h then
            targetH = torch.round(h / w * targetW)
        else
            targetW = torch.round(w / h * targetH)
        end

        return image.scale(input, targetW, targetH, 'bicubic'), image.scale(label, targetW, targetH, 'bicubic')
    end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
function M.RandomSizedCrop(size, minFrac)
    local scale = M.Scale(size)
    local crop = M.CenterCrop(size)

    return function(input, label)
        local attempt = 0
        repeat
            local area = input:size(2) * input:size(3)
            minFrac = minFrac or 0.08
            local targetArea = torch.uniform(minFrac, 1.0) * area

            local aspectRatio = torch.uniform(3/4, 4/3)
            local w = torch.round(math.sqrt(targetArea * aspectRatio))
            local h = torch.round(math.sqrt(targetArea / aspectRatio))

            if torch.uniform() < 0.5 then
                w, h = h, w
            end

            if h <= input:size(2) and w <= input:size(3) then
                local y1 = torch.random(0, input:size(2) - h)
                local x1 = torch.random(0, input:size(3) - w)

                local out = image.crop(input, x1, y1, x1 + w, y1 + h)
                local out_label = image.crop(label, x1, y1, x1 + w, y1 + h)
                assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

                return image.scale(out, size, size, 'bicubic'), image.scale(out_label, size, size, 'bicubic')
            end
            attempt = attempt + 1
        until attempt >= 10

        -- fallback
        return crop(scale(input)), crop(scale(label))
    end
end


function M.Rotation(deg)
    return function(input, label)
        if deg ~= 0 then
            local rot = (torch.uniform() - 0.5) * deg * math.pi / 180
            input = image.rotate(input, rot, 'bilinear')
            label = image.rotate(label, rot, 'bilinear')
        end
        return input, label
    end
end

function M.HorizontalFlip(prob)
    return function(input, label)
        if torch.uniform() < prob then
            input = image.hflip(input)
            label = image.hflip(label)
        end
        return input, label
    end
end

function M.VerticalFlip(prob)
    return function(input, label)
        if torch.uniform() < prob then
            input = image.vflip(input)
            label = image.vflip(label)
        end
        return input, label
    end
end


--- [[Lighting Noise]] ---

local function blend(img1, img2, alpha)
return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
    dst:resizeAs(img)
    dst[1]:zero()
    dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
    dst[2]:copy(dst[1])
    dst[3]:copy(dst[1])
    return dst
end


function M.Saturation(var)
    local gs

    return function(input)
        gs = gs or input.new()
        grayscale(gs, input)

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(input, gs, alpha)
        return input
    end
end

function M.Brightness(var)
    local gs

    return function(input)
        gs = gs or input.new()
        gs:resizeAs(input):zero()

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(input, gs, alpha)
        return input
    end
end

function M.Contrast(var)
    local gs

    return function(input)
        gs = gs or input.new()
        grayscale(gs, input)
        gs:fill(gs[1]:mean())

        local alpha = 1.0 + torch.uniform(-var, var)
        blend(input, gs, alpha)
        return input
    end
end


function M.RandomOrder(ts)
    return function(input, label)
        local img = input.img or input

        local BW = false
        if img:size(1) == 1 then  -- add dummy channels
            local img2 = torch.Tensor(3,img:size(2),img:size(3))
            img2[1]:copy(img[1])
            img2[2]:copy(img[1])
            img2[2]:copy(img[1])
            img = img2
            BW = true
        end

        local order = torch.randperm(#ts)
        for i=1,#ts do
            img = ts[order[i]](img)
        end

        if BW == true then
            out = torch.Tensor(1,img:size(2),img:size(3))
            out[1]:copy(img[1])
            return out, label
        end
        return img, label
    end
end

function M.IntesityJitter(opt)
    local brightness = opt.brightness or 0
    local contrast = opt.contrast or 0
    local saturation = opt.saturation or 0

    local ts = {}
    if brightness ~= 0 then
        table.insert(ts, M.Brightness(brightness))
    end
    if contrast ~= 0 then
        table.insert(ts, M.Contrast(contrast))
    end
    if saturation ~= 0 then
        table.insert(ts, M.Saturation(saturation))
    end

    if #ts == 0 then
        return function(input) return input end
    end

    return M.RandomOrder(ts)
end

return M
