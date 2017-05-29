local ORAlign, parent = torch.class('nn.ORAlign', 'nn.Module')

function ORAlign:__init(nOrientation)
    parent.__init(self)
    self.nOrientation = nOrientation
    self.mainDirection = torch.Tensor()
    self.initFlag = false
end

function ORAlign:lazyInit(input)
    if input:nDimension() == 3 then
        self.nBatch = 1
        self.nFeature = input:size(1) / self.nOrientation
    elseif input:nDimension() == 4 then
        self.nBatch = input:size(1)
        self.nFeature = input:size(2) / self.nOrientation
    end
    self.mainDirection:resize(self.nBatch, self.nFeature)
    self.output:resizeAs(input)
    self.gradInput:resizeAs(input)
    self.initFlag = true
end

local function makeContiguous(self, input, gradOutput)
    if not input:isContiguous() then
        self._input = self._input or input.new()
        self._input:resizeAs(input):copy(input)
        input = self._input
    end
    if gradOutput and (not gradOutput:isContiguous()) then
        self._gradOutput = self._gradOutput or gradOutput.new()
        self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
        gradOutput = self._gradOutput
    end
    return input, gradOutput
end

function ORAlign:updateOutput(input)
    if not self.initFlag then self:lazyInit(input) end
    if (input:nDimension() == 3 and self.nBatch ~= 1) or 
       (input:nDimension() == 4 and self.nBatch ~= input:size(1)) then 
        self:lazyInit(input)
    end
    input = makeContiguous(self, input)
    input.orn.RIE_AlignFeature(
        self,
        input, 
        self.mainDirection, 
        self.output, 
        self.nBatch, 
        self.nFeature, 
        self.nOrientation
    )

    return self.output
end

function ORAlign:updateGradInput(input, gradOutput)
    input, gradOutput = makeContiguous(self, input, gradOutput)
    input.orn.RIE_UnAlignFeature(
        self,
        self.gradInput, 
        self.mainDirection, 
        gradOutput, 
        self.nBatch, 
        self.nFeature, 
        self.nOrientation
    )

    return self.gradInput
end
