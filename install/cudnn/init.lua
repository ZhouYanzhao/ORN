require 'torch'
require 'cudnn'
require 'cuorn'
require 'libcudnnorn'

include('ORConv.lua')

-- monkey patch BN
local BN = cudnn.SpatialBatchNormalization
function BN:createIODescriptors(input)
   assert(input:dim() == self.nDim)
   assert(torch.typename(self.weight) == 'torch.CudaTensor' and torch.typename(self.bias) == 'torch.CudaTensor',
          'Only CUDA tensors are supported for cudnn.BatchNormalization!')
   if not self.iDesc or not self.oDesc or not input:isSize(self.iSize) then
      local nFeature = self.running_mean:numel()
      self.iSize = input:size()
      self.output:resizeAs(input)
      if self.nDim == 4 and self.iSize[2] > nFeature then
         local tmp = input.new():resize(
            self.iSize[1], 
            nFeature,
            self.iSize[3] * (self.iSize[2] / nFeature),
            self.iSize[4]
         )
         self.iDesc = cudnn.toDescriptor(tmp)
         self.oDesc = cudnn.toDescriptor(tmp)
      else
         self.iDesc = cudnn.toDescriptor(input)
         self.oDesc = cudnn.toDescriptor(self.output)
      end
      local biasSize = torch.ones(self.nDim):totable()
      biasSize[2] = nFeature
      self.sDesc = cudnn.toDescriptor(self.bias:view(table.unpack(biasSize)))
   end
end

-- monkey patch cudnn.convert
local layer_list = {
  'BatchNormalization',
  'SpatialBatchNormalization',
  'SpatialConvolution',
  'SpatialCrossMapLRN',
  'SpatialFullConvolution',
  'SpatialMaxPooling',
  'SpatialAveragePooling',
  'ReLU',
  'Tanh',
  'Sigmoid',
  'SoftMax',
  'LogSoftMax',
  'VolumetricBatchNormalization',
  'VolumetricConvolution',
  'VolumetricMaxPooling',
  'VolumetricAveragePooling',
  'ORConv'
}

function cudnn.convert(net, dst, exclusion_fn)
  return net:replace(function(x)
    local y = 0
    local src = dst == nn and cudnn or nn
    local src_prefix = src == nn and 'nn.' or 'cudnn.'
    local dst_prefix = dst == nn and 'nn.' or 'cudnn.'

    local function convert(v)
      local y = {}
      torch.setmetatable(y, dst_prefix..v)
      if v == 'ReLU' then y = dst.ReLU() end -- because parameters
      for k,u in pairs(x) do y[k] = u end
      if src == cudnn and x.clearDesc then x.clearDesc(y) end
      if src == cudnn and v == 'SpatialAveragePooling' then
        y.divide = true
        y.count_include_pad = v.mode == 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
      end
      return y
    end

    if exclusion_fn and exclusion_fn(x) then
      return x
    end
    local t = torch.typename(x)
    if t == 'nn.SpatialConvolutionMM' then
      y = convert('SpatialConvolution')
    elseif t == 'inn.SpatialCrossResponseNormalization' then
      y = convert('SpatialCrossMapLRN')
    else
      for i,v in ipairs(layer_list) do
        if torch.typename(x) == src_prefix..v then
          y = convert(v)
        end
      end
    end
    return y == 0 and x or y
  end)
end