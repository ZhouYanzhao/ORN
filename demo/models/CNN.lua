require 'nn'
local utils = paths.dofile '../utils.lua'

return function (opt)
    -- Conventional CNN
    local model = nn.Sequential()
    -- feature learning
    model:add(nn.SpatialConvolutionMM(1, 80, 3, 3))
    model:add(nn.ReLU())             
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 

    model:add(nn.SpatialConvolutionMM(80, 160, 3, 3))
    model:add(nn.ReLU())            
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.SpatialConvolutionMM(160, 320, 3, 3, 1, 1, 1, 1))
    model:add(nn.ReLU())            
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.SpatialConvolutionMM(320, 640, 3, 3))
    model:add(nn.ReLU())

    -- classifier
    model:add(nn.View(640))              
    model:add(nn.Linear(640, 1024)) 
    model:add(nn.ReLU())    
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(1024, 10))     

    return model
end