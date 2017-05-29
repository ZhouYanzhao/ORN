require 'nn'
local utils = paths.dofile '../utils.lua'

return function (opt)
    assert(opt.orientation, 'missing orientation')
    -- Oriented Response Network
    local model = nn.Sequential()

    -- extend input to an omnidirectional tensor map
    model:add(nn.View(-1, 32, 32))
    model:add(nn.Replicate(opt.orientation, 2))

    -- feature learning
    model:add(nn.ORConv(1, 10, opt.orientation, 3, 3))
    model:add(nn.ReLU())             
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) 

    model:add(nn.ORConv(10, 20, opt.orientation, 3, 3))
    model:add(nn.ReLU())            
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.ORConv(20, 40, opt.orientation, 3, 3, 1, 1, 1, 1))
    model:add(nn.ReLU())            
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    model:add(nn.ORConv(40, 80, opt.orientation, 3, 3))
    model:add(nn.ReLU())
    
    -- rotation invariant encoding
    local nFeatureDim = nil
    if opt.useORAlign then
        -- ORAlign
        model:add(nn.ORAlign(opt.orientation))
        nFeatureDim = 80 * opt.orientation
    elseif opt.useORPooling then
        -- ORPooling
        model:add(nn.View(80, opt.orientation))
        model:add(nn.Max(2, 2))
        nFeatureDim = 80
    else
        -- None
        nFeatureDim = 80 * opt.orientation
    end

    -- classifier
    model:add(nn.View(nFeatureDim))
    model:add(nn.Linear(nFeatureDim, 1024))
    model:add(nn.ReLU())    
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(1024, 10))     
    
    return model
end
