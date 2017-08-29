local THNN = require 'nn.THNN'
local ORConv, parent = torch.class('nn.ORConv', 'nn.Module')

function ORConv:__init(nInputPlane, nOutputPlane, ARF, kW, kH, dW, dH, padW, padH)
    parent.__init(self)

    self.nInputPlane = nInputPlane
    self.nOutputPlane = nOutputPlane
    if type(ARF) == 'table' then
        local n = table.getn(ARF)
        assert(n > 0, "empty ARF Parameters")
        if n == 1 then
            self.nOrientation, self.nRotation = ARF[1], ARF[1]
        else
            self.nOrientation, self.nRotation = ARF[1], ARF[2]
        end
    else
        self.nOrientation, self.nRotation = ARF, ARF
    end
    assert(math.log(self.nOrientation) % math.log(2) == 0, 'invalid nOrientation')
    assert(math.log(self.nRotation) % math.log(2) == 0, 'invalid nRotation')
    self.kW = kW or 3
    self.kH = kH or self.kW
    self.dW = dW or 1
    self.dH = dH or self.dW
    self.padW = padW or 0
    self.padH = padH or self.padW

    self.weight = torch.Tensor(nOutputPlane, nInputPlane, self.nOrientation, self.kH, self.kW)
    self.gradWeight = self.weight:clone():zero()

    self.bias = torch.Tensor(nOutputPlane * self.nRotation)
    self.gradBias = self.bias:clone():zero()

    self:createInitFuncs()
    self.mode = self:checkMode()
    self.initFuncs[self.mode]()

    self:reset()
end

function ORConv:createInitFuncs()
    self.initFuncs = {
        fast = function ()
            self:clearState()

            self.bW, self.bH = self.kW, self.kH
            self.weightBufferSize = {self.nOutputPlane * self.nRotation, self.nInputPlane * self.nOrientation * self.bH * self.bW}
        
            self.ARFRotate = function (module, input)
                module.indices = module.indices or module:getMappingIndices():typeAs(input)
                input.orn.ARF_MappingRotate(
                    module,
                    module.weight,
                    module.indices,
                    module.weightBuffer,
                    module.kW, module.kH,
                    module.nInputPlane, module.nOutputPlane,
                    module.nOrientation, module.nRotation
                )
            end
            self.ARFAlign = function (module, input)
                input.orn.ARF_MappingAlign(
                    module,
                    module.gradWeight,
                    module.indices,
                    module.gradWeightBuffer,
                    module.kW, module.kH, 
                    module.nInputPlane, module.nOutputPlane, 
                    module.nOrientation, module.nRotation
                )
            end
        end,
        general = function (extension)
            self:clearState()
            self.interpolation = 'bilinear'
            if extension == 'CirclePadding' then
                local radius = 0.5 * torch.sqrt(2) * ((self.kW > self.kH and self.kW or self.kH) - 1)
                self.bW = torch.ceil(radius) * 2 + 1
                self.bH = self.bW
            else
                self.bW = self.kW > self.kH and self.kW or self.kH
                self.bH = self.bW
            end
            self.weightBufferSize = {self.nOutputPlane * self.nRotation, self.nInputPlane * self.nOrientation * self.bH * self.bW}
            
            self.ARFRotate = function (module, input)
                module.buffer = module.buffer or input.new():resize(table.unpack(module.weightBufferSize))
                module.indices = module.indices or module:getInterpolationIndices('clockwise'):typeAs(input)
                module.invIndices = module.invIndices or module:getInterpolationIndices('counter-clockwise'):typeAs(input)
                module.factors = module.factors or module:getDFTFactors(module.nOrientation, module.nRotation, 'clockwise'):typeAs(input)
                module.invFactors = module.invFactors or module:getDFTFactors(module.nOrientation, module.nRotation, 'counter-clockwise'):typeAs(input)
                input.orn.ARF_Rotate(
                    module,
                    module.weight,
                    module.indices,
                    module.factors,
                    module.weightBuffer,
                    module.buffer,
                    module.kW, module.kH,
                    module.bW, module.bH,
                    module.nInputPlane, module.nOutputPlane,
                    module.nOrientation, module.nRotation
                )
            end
            self.ARFAlign = function (module, input)
                input.orn.ARF_Align(
                    module,
                    module.gradWeight,
                    module.invIndices,
                    module.invFactors,
                    module.buffer,
                    module.gradWeightBuffer,
                    module.kW, module.kH,
                    module.bW, module.bH,
                    module.nInputPlane, module.nOutputPlane, 
                    module.nOrientation, module.nRotation
                )
            end
        end
    }
end

function ORConv:checkMode()
    if (self.kW == self.kH) and 
        (self.kW == 1 or self.kW == 3) and
        (self.nOrientation == 1 or self.nOrientation == 4 or self.nOrientation == 8) and
        (self.nRotation == 1 or self.nRotation == 4 or self.nRotation == 8) and
        (self.nOrientation == 1 or self.nRotation <= self.nOrientation) then
        return 'fast'
    else
        return 'general'
    end
end

function ORConv:getMappingIndices()
    local kernelIndices = {
        -- 1x1
        [1] = {
            [0] = {1},
            [45] = {1},
            [90] = {1},
            [135] = {1},
            [180] = {1},
            [225] = {1},
            [270] = {1},
            [315] = {1}
        },
        -- 3x3
        [3] = {
            [0] = {1,2,3,4,5,6,7,8,9},
            [45] = {2,3,6,1,5,9,4,7,8},
            [90] = {3,6,9,2,5,8,1,4,7},
            [135] = {6,9,8,3,5,7,2,1,4},
            [180] = {9,8,7,6,5,4,3,2,1},
            [225] = {8,7,4,9,5,1,6,3,2},
            [270] = {7,4,1,8,5,2,9,6,3},
            [315] = {4,1,2,7,5,3,8,9,6}
        }
    }
    local deltaOrientation = 360 / self.nOrientation
    local deltaRotation = 360 / self.nRotation
    local indices = torch.Tensor(self.nOrientation * self.kH * self.kW, self.nRotation)
    for i = 1, self.nOrientation do
        for j = 1, self.kH * self.kW do
            for k = 1, self.nRotation do
                local angle = deltaRotation * (k - 1)
                local layer = (i - 1 + torch.floor(angle / deltaOrientation)) % self.nOrientation + 1
                local kernel = kernelIndices[self.kW][angle][j]
                indices[{(i - 1) * self.kH * self.kW + j, k}] = (layer - 1) * self.kH * self.kW + kernel
            end
        end
    end
    return indices
end

local function between(value, lowerBound, higherBound)
    return (value >= lowerBound) and (value <= higherBound)
end

function ORConv:getInterpolationIndices(direction)
    local srcW
    local srcH
    local dir
    if direction == 'clockwise' then
        srcW, srcH = self.kW, self.kH
        dstW, dstH = self.bW, self.bH
        dir = -1
    else
        srcW, srcH = self.bW, self.bH
        dstW, dstH = self.kW, self.kH
        dir = 1
    end

    local indices = torch.Tensor(self.nRotation, dstH, dstW, 8)
    local cx = (dstW - 1) * 0.5
    local cy = (dstH - 1) * 0.5
    local theta
    for k = 0, self.nRotation - 1 do
        theta = dir * k * 2*math.pi / self.nRotation
        local T = torch.Tensor({
            {math.cos(theta), -math.sin(theta), cx - math.cos(theta) * cx + math.sin(theta) * cy}, 
            {math.sin(theta), math.cos(theta), cy - math.sin(theta) * cx - math.cos(theta) * cy}, 
            {0, 0, 1}
        })
        for i = 0, dstW - 1 do
            for j = 0, dstH - 1 do
                local P = (T * torch.Tensor({{i, j, 1}}):t()):view(-1)
                        - torch.Tensor({0.5 * (dstW - srcW), 0.5 * (dstH - srcH), 0})
                local u = math.floor(P[1] + 0.001)
                local v = math.floor(P[2] + 0.001)
                local a = P[1] - u
                local b = P[2] - v
                if self.interpolation == 'nearest' then
                    u = u + torch.round(a)
                    v = v + torch.round(b)
                    if between(u, 0, srcW - 1) and between(v, 0, srcH - 1) then
                        indices[k + 1][j + 1][i + 1][1] = 1
                        indices[k + 1][j + 1][i + 1][2] = v * srcW + u
                        indices[k + 1][j + 1][i + 1][3] = 0
                        indices[k + 1][j + 1][i + 1][4] = 0
                        indices[k + 1][j + 1][i + 1][5] = 0
                        indices[k + 1][j + 1][i + 1][6] = 0
                        indices[k + 1][j + 1][i + 1][7] = 0
                        indices[k + 1][j + 1][i + 1][8] = 0
                    else
                        indices[k + 1][j + 1][i + 1][1] = 0
                        indices[k + 1][j + 1][i + 1][2] = 0
                        indices[k + 1][j + 1][i + 1][3] = 0
                        indices[k + 1][j + 1][i + 1][4] = 0
                        indices[k + 1][j + 1][i + 1][5] = 0
                        indices[k + 1][j + 1][i + 1][6] = 0
                        indices[k + 1][j + 1][i + 1][7] = 0
                        indices[k + 1][j + 1][i + 1][8] = 0
                    end
                else 
                    if between(u, 0, srcW - 1) and between(v, 0, srcH - 1) then
                        indices[k + 1][j + 1][i + 1][1] = (1 - b) * (1 - a)
                        indices[k + 1][j + 1][i + 1][2] = v * srcW + u
                    else
                        indices[k + 1][j + 1][i + 1][1] = 0
                        indices[k + 1][j + 1][i + 1][2] = 0
                    end
                    if between(u + 1, 0, srcW - 1) and between(v, 0, srcH - 1) then
                        indices[k + 1][j + 1][i + 1][3] = (1 - b) * (a)
                        indices[k + 1][j + 1][i + 1][4] = v * srcW + (u + 1)
                    else
                        indices[k + 1][j + 1][i + 1][3] = 0
                        indices[k + 1][j + 1][i + 1][4] = 0
                    end
                    if between(u, 0, srcW - 1) and between(v + 1, 0, srcH - 1) then
                        indices[k + 1][j + 1][i + 1][5] = (b) * (1 - a)
                        indices[k + 1][j + 1][i + 1][6] = (v + 1) * srcW + u
                    else
                        indices[k + 1][j + 1][i + 1][5] = 0
                        indices[k + 1][j + 1][i + 1][6] = 0
                    end
                    if between(u + 1, 0, srcW - 1) and between(v + 1, 0, srcH - 1) then
                        indices[k + 1][j + 1][i + 1][7] = (b) * (a)
                        indices[k + 1][j + 1][i + 1][8] = (v + 1) * srcW + (u + 1)
                    else
                        indices[k + 1][j + 1][i + 1][7] = 0
                        indices[k + 1][j + 1][i + 1][8] = 0
                    end
                end
            end
        end
    end
    return indices
end

function ORConv:getDFTFactors(nOrientation, nRotation, direction)
    if nOrientation == 1 then
        return torch.Tensor()
    end

    local freqs = torch.Tensor(nOrientation)
    freqs[{{1, nOrientation / 2}}] = torch.range(0, nOrientation / 2 - 1)
    freqs[{{nOrientation / 2 + 1, nOrientation}}] = torch.range(-nOrientation / 2, -1)

    local DFTs = torch.Tensor(nRotation, nOrientation, nOrientation)
    local theta
    for k = 0, nRotation - 1 do
        theta = (direction == 'clockwise' and k or nRotation - k) * 2 * math.pi / nRotation
        for m = 0, nOrientation - 1 do
            for n = 0, nOrientation - 1 do
                local sum = 0
                for l = 0, nOrientation - 1 do
                    sum = sum + math.cos(l * 2*math.pi / nOrientation * (m - n) - freqs[l + 1] * theta)
                end
                DFTs[k + 1][m + 1][n + 1] = sum / nOrientation
            end
        end
    end
    return DFTs
end

function ORConv:general(extension)
    self.mode = 'general'
    self.initFuncs[self.mode](extension)
    return self
end

function ORConv:noBias()
    self.bias = nil
    self.gradBias = nil
    return self
end

function ORConv:reset()
    local n = self.kW * self.kH * self.nInputPlane * self.nOrientation
    self.weight:normal(0, math.sqrt(2 / n))
    if self.bias then self.bias:zero() end
end

function ORConv:updateOutput(input)
    assert(input.THNN, torch.type(input)..'.THNN backend not imported')
    self.finput = self.finput or input.new()
    self.fgradInput = self.fgradInput or input.new()
    -- backward compatibility
    if self.padding then
        self.padW = self.padding
        self.padH = self.padding
        self.padding = nil
    end
    
    -- generate rotated versions of ARF on the fly
    self.weightBuffer = self.weightBuffer or input.new():resize(table.unpack(self.weightBufferSize))
    self.gradWeightBuffer = self.gradWeightBuffer or input.new():resize(table.unpack(self.weightBufferSize))
    self.ARFRotate(self, input)

    input.THNN.SpatialConvolutionMM_updateOutput(
        input:cdata(),
        self.output:cdata(),
        self.weightBuffer:cdata(),
        THNN.optionalTensor(self.bias),
        self.finput:cdata(),
        self.fgradInput:cdata(),
        self.bW, self.bH,
        self.dW, self.dH,
        self.padW, self.padH
    )
    return self.output
end

function ORConv:updateGradInput(input, gradOutput)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   if self.gradInput then
        input.THNN.SpatialConvolutionMM_updateGradInput(
            input:cdata(),
            gradOutput:cdata(),
            self.gradInput:cdata(),
            self.weightBuffer:cdata(),
            self.finput:cdata(),
            self.fgradInput:cdata(),
            self.bW, self.bH,
            self.dW, self.dH,
            self.padW, self.padH
        )
        return self.gradInput
   end
end

function ORConv:accGradParameters(input, gradOutput, scale)
    assert(input.THNN, torch.type(input)..'.THNN backend not imported')
    scale = scale or 1
    assert((self.bias and self.gradBias) or (self.bias == nil and self.gradBias == nil))
    self.gradWeightBuffer:zero()
    input.THNN.SpatialConvolutionMM_accGradParameters(
        input:cdata(),
        gradOutput:cdata(),
        self.gradWeightBuffer:cdata(),
        THNN.optionalTensor(self.gradBias),
        self.finput:cdata(),
        self.fgradInput:cdata(),
        self.bW, self.bH,
        self.dW, self.dH,
        self.padW, self.padH,
        scale
    )

    self.ARFAlign(self, input)
end

function ORConv:type(type,tensorCache)
    self.finput = self.finput and torch.Tensor()
    self.fgradInput = self.fgradInput and torch.Tensor()
    return parent.type(self,type,tensorCache)
end

function ORConv:__tostring__()
    local ARF = self.nOrientation == self.nRotation and string.format('[%d]', self.nOrientation) or string.format("[%d-%d]", self.nOrientation, self.nRotation)
    local s = string.format('%s(%s %d -> %d, %dx%d', torch.type(self),
    ARF, self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
    if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
        s = s .. string.format(', %d,%d', self.dW, self.dH)
    end
    if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
        s = s .. ', ' .. self.padW .. ',' .. self.padH
    end
    local interpolation = self.interpolation == 'nearest' and ' (nearest)' or ''
    if self.bias then
        return s .. '), ' .. self.mode .. ' mode' .. interpolation
    else
        return s .. ') without bias, ' .. self.mode .. ' mode' .. interpolation
    end
end

function ORConv:clearState()
    nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
    self.weightBuffer = nil
    self.gradWeightBuffer = nil
    self.buffer = nil
    self.indices = nil
    self.invIndices = nil
    self.freqs = nil
    collectgarbage()
    return parent.clearState(self)
end
