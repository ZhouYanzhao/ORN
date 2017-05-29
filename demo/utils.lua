require 'optim'
require 'cudnn'
local tnt = require 'torchnet'
local optnet = require 'optnet'
local c = require 'trepl.colorize'
local argcheck = require 'argcheck'

local utils = {}

-- console colors
utils.colors = {
    info = c.blue,
    warning = c.yellow,
    danger = c.red,
    success = c.green,
    default = c.white
}

function utils.parseOpt(opt)
    opt = xlua.envparams(opt)
    opt.epochStep = tonumber(opt.epochStep) or loadstring('return '..opt.epochStep)() 
    opt.gpuDevice = tonumber(opt.gpuDevice) or loadstring('return '..opt.gpuDevice)() 
    opt = tablex.merge(opt, loadstring('return '..opt.customParams)(), true)
    return opt
end

function utils.makeDataParallelTable(model, gpus)
    local fastest, benchmark = cudnn.fastest, cudnn.benchmark

    local dpt = nn.DataParallelTable(1, true, true)
     :add(model, gpus)
     :threads(function()
        local cudnn = require 'cudnn'
        require 'cudnnorn'
        cudnn.fastest, cudnn.benchmark = fastest, benchmark
     end)
    dpt.gradInput = nil

    model = dpt:cuda()
   return model
end

function utils.cast(data, mode)
    if mode == 'cpu' then
        return data:float()
    else
        return data:cuda()
    end
end

local logtext = argcheck{
    noordered = true,
    {name = 'filename', type = 'string', opt = true},
    {name = 'keys', type = 'table'},
    {name = 'format', type = 'table', opt = true},
    {name = 'separator', type = 'string', default = ' | '},
    {name = 'append', type = 'boolean', default = false},
    call =
    function(filename, keys__, format__, separator, append)
        local keys = {}
        for idx, key in ipairs(keys__) do
            local format = format__ and format__[idx]
            if not format then
                table.insert(keys, {name=key, format=function(value) return string.format('%s %s', key, value) end})
            elseif type(format) == 'function' then
                table.insert(keys, {name=key, format=format})
            elseif type(format) == 'string' then
                table.insert(keys, {name=key, format=function(value) return string.format(format, value) end})
            else
                error('format must be a string or a function')
            end
        end
        if filename and not append then
            local f = io.open(filename, 'w') -- reset the file
            assert(f, string.format('could not open file <%s> for writing', filename))
            f:close()
        end
        return function(log)
            local txt = {}
            for _, key in ipairs(keys) do
                local format = key.format(log:get(key.name))
                assert(type(format) == 'string', string.format('value for key %s cannot be converted to string', key))
                table.insert(txt, format)
            end
            txt = table.concat(txt, separator)
            if filename then
                local f = io.open(filename, 'a+') -- append
                assert(f, string.format('could not open file <%s> for writing', filename))
                f:write(txt)
                f:write('\n')
                f:close()
            else
                if log.notify then
                    print('{{MSG:' .. txt .. '}}')
                else
                    print(txt)
                end
            end
        end
    end
}

local logstatus = argcheck{
    noordered=true,
    {name = 'filename', type = 'string', opt = true},
    {name = 'append', type = 'boolean', default = false},
    call =
    function(filename, append)
        if filename and not append then
            local f = io.open(filename, 'w') -- reset the file
            assert(f, string.format('could not open file <%s> for writing', filename))
            f:close()
        end
        return function(log, key, value)
            if key == '__status__' then
                local status = tostring(value)
                if filename then
                    local f = io.open(filename, 'a+') -- append
                    assert(f, string.format('could not open file <%s> for writing', filename))
                    f:write(status)
                    f:write('\n')
                    f:close()
                else
                    if log.notify then
                        print('{{MSG:' .. status .. '}}')
                    else
                        print(status)
                    end
                end
            end 
        end
    end
}

function utils.getLogger(opt)
    if opt.removeOldCheckpoints then
        os.execute('rm ' .. paths.concat(opt.savePath, '*.t7'))
    end
    paths.mkdir(opt.savePath)
    local logKeys = loadstring('return '..opt.logKeys)()
    local logFormats = loadstring('return '..opt.logFormats)()
    for idx = 1, table.getn(logKeys) do
        logFormats[idx] = logKeys[idx] .. ' ' .. logFormats[idx]
    end

    return tnt.Log{
        keys = logKeys,
        onFlush = {
            -- print on screen
            logtext{
                keys = logKeys,
                format = logFormats
            },
            -- save to file
            opt.enableSave and logtext{
                filename = paths.concat(opt.savePath, 'log.txt'),
                keys = logKeys,
                format = logFormats,
                append = true
            } or nil
        },
        onSet = {
            -- print status to screen
            logstatus{},
            -- save to file
            opt.enableSave and logstatus{
                filename = paths.concat(opt.savePath, 'log.txt'),
                append = true
            } or nil
        }
    }
end

function utils.equalSize(A, B)
   if A:nDimension() == B:nDimension() then
      for i = 1, A:nDimension() do
         if A:size(i) ~= B:size(i) then
            return false
         end
      end
      return true
   end
   return false
end

function utils.tableToString(tab)
    local content = {}
    for k, v in pairs(tab) do
       table.insert(content, string.format('%s: %s', k, v))
    end
    return table.concat(content, '\r\n')
end

function utils.testModel(opt, model)
    local sample = opt.customSampleSize and torch.Tensor(torch.LongStorage(opt.customSampleSize)) or torch.Tensor(10, opt.imageChannel, opt.imageSize, opt.imageSize)
    local target = opt.customTargetSize and torch.Tensor(torch.LongStorage(opt.customTargetSize)) or torch.Tensor(10, opt.numClasses)
    sample = utils.cast(sample, opt.trainMode)
    if opt.optnetOptimize then 
        optnet.optimizeMemory(model, sample, {inplace=true, mode='training'})
    end
    local params, gradParams = model:getParameters()
    local info = {
        params = params:numel(),
    }
    return utils.equalSize(model:forward(sample), target), info
end

function utils.train(opt, model, dataset, logger)
    local engine = tnt.OptimEngine()
    local criterion = utils.cast(nn.CrossEntropyCriterion(), opt.trainMode)
    local avgMeter = tnt.AverageValueMeter()
    local clsMeter = tnt.ClassErrorMeter{topk = {1}}
    local timeMeter = tnt.TimeMeter()
    local trainIterator = dataset('train')
    local validationIterator = dataset('validation')
    local testIterator = dataset('test')
    local bestValidationAcc = -1
    local currentAcc = -1
    local bestSavePath = ''

    -- add hooks
    engine.hooks.onStartEpoch = function(state)
        avgMeter:reset()
        clsMeter:reset()
        timeMeter:reset()
        
        local epoch = state.epoch + 1
        if torch.type(opt.epochStep) == 'number' and epoch % opt.epochStep == 0 or
          torch.type(opt.epochStep) == 'table' and tablex.find(opt.epochStep, epoch) then
            opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
            state.config = tablex.deepcopy(opt)
            state.optim = tablex.deepcopy(opt)
        end
    end

    engine.hooks.onEndEpoch = function(state)
        local improved = false
        
        logger:set{
            trainLoss = avgMeter:value(),
            trainAcc = 100 - clsMeter:value{k = 1}
        }

        -- validation
        if validationIterator and state.epoch >= opt.evaluateEpoch then
            avgMeter:reset()
            clsMeter:reset()

            engine:test{
                network = state.network,
                iterator = validationIterator,
                criterion = criterion,
            }

            currentAcc = 100 - clsMeter:value{k = 1}
            logger:set{
                testLoss = avgMeter:value(),
                testAcc = currentAcc
            }

            improved =  currentAcc > bestValidationAcc
            if improved then 
                bestValidationAcc = currentAcc
            end
        else
            logger:set{
                testLoss = -1,
                testAcc = -1
            }
        end
        -- create checkpoint
        if  opt.enableSave and
            ((opt.checkpoint == 'better' and improved) or
            (opt.checkpoint == 'all') or 
            (tonumber(opt.checkpoint) and state.epoch % opt.checkpoint == 0)) then
            bestSavePath = paths.concat(opt.savePath, string.format('%d-%.2f-%s.t7', state.epoch, currentAcc, os.date('%Y%m%d%H%M%S')))
            logger.notify = false
            logger:status('checkpoint saved at ' .. bestSavePath)
            torch.save(bestSavePath, state.network:clearState())
            collectgarbage()
        end
        -- send notification 
        logger.notify = 
            (opt.notify == 'better' and improved) or 
            (opt.notify == 'all') or 
            (tonumber(opt.notify) and state.epoch % opt.notify == 0)
        -- time estimate
        local elapse = timeMeter:value()
        local remain = (state.maxepoch - state.epoch) * elapse
        logger:set{
            epoch = state.epoch,
            elapse = elapse,
            note = opt.note or 'Exp',
            remain = string.format('%2d:%2d:%2d', remain / 60 / 60, remain / 60 % 60, remain % 60)
        }

        logger:flush()
    end

    engine.hooks.onForwardCriterion = function(state)
        avgMeter:add(state.criterion.output)
        clsMeter:add(state.network.output, state.sample.target)
    end

    -- create buffer
    local inputs = utils.cast(torch.Tensor(), opt.trainMode)
    local targets = utils.cast(torch.Tensor(), opt.trainMode)

    engine.hooks.onSample = function(state)
        inputs:resize(state.sample.input:size()):copy(state.sample.input)
        targets:resize(state.sample.target:size()):copy(state.sample.target)
        state.sample.input = inputs
        state.sample.target = targets
    end

    engine.hooks.onEnd = function(state)
        if state.training then
            local finalAcc = -1
            if testIterator and opt.enableSave then
                avgMeter:reset()
                clsMeter:reset()
                -- load best model
                state.network = nil
                collectgarbage()
                state.network = torch.load(bestSavePath)
                -- final test
                engine:test{
                    network = state.network,
                    iterator = testIterator, 
                    criterion = criterion,
                }
                finalAcc = 100 - clsMeter:value{k = 1}
            end
            logger:status(string.format('best model: %s', bestSavePath))
            logger.notify = true
            logger:status(string.format('[%s] all done. bestValidationAcc: %.2f, finalTestAcc: %.2f', opt.note, bestValidationAcc, finalAcc))
        end
    end
    
    collectgarbage()
    
    -- start training
    engine:train{
        network = model,
        iterator = trainIterator,
        criterion = criterion,
        optimMethod = loadstring('return optim.' .. opt.optimMethod)(),
        config = tablex.deepcopy(opt),
        maxepoch = opt.maxEpoch,
    }
end

function utils.test(opt, model, dataset, logger)
    local engine = tnt.OptimEngine()
    local criterion = utils.cast(nn.CrossEntropyCriterion(), opt.trainMode)
    local clsMeter = tnt.ClassErrorMeter{topk = {1}}
    local timeMeter = tnt.TimeMeter()
    local validationIterator = dataset('validation')
    local testIterator = dataset('test')
    -- create buffer
    local inputs = utils.cast(torch.Tensor(), opt.trainMode)
    local targets = utils.cast(torch.Tensor(), opt.trainMode)
 
    -- add hooks
    engine.hooks.onForwardCriterion = function(state)
        clsMeter:add(state.network.output, state.sample.target)
    end

    engine.hooks.onSample = function(state)
        inputs:resize(state.sample.input:size()):copy(state.sample.input)
        targets:resize(state.sample.target:size()):copy(state.sample.target)
        state.sample.input = inputs
        state.sample.target = targets
    end

    timeMeter:reset()
    clsMeter:reset()
    engine:test{
        network = model,
        iterator = validationIterator,
        criterion = criterion,
    }
    local validationAcc = 100 - clsMeter:value{k = 1}
    clsMeter:reset()
    engine:test{
        network = model,
        iterator = testIterator,
        criterion = criterion,
    }
    local testAcc = 100 - clsMeter:value{k = 1}
    local elapse = string.format('%2d:%2d:%2d', timeMeter:value() / 60 / 60, timeMeter:value() / 60 % 60, timeMeter:value() % 60)

    logger.notify = true
    logger:status(string.format('[%s] elapse: %s, validationAcc: %.2f, testAcc: %.2f', opt.note, elapse, validationAcc, testAcc))
end

return utils 