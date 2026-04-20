clc; clear;

p = furuta_params();
env = furuta_env();
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Actor
actorNet = [
    featureInputLayer(4)
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(1)
    tanhLayer];

actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

% Critic
statePath  = featureInputLayer(4,'Name','state');
actionPath = featureInputLayer(1,'Name','action');

criticNet = layerGraph();
criticNet = addLayers(criticNet,statePath);
criticNet = addLayers(criticNet,actionPath);
criticNet = addLayers(criticNet,[
    concatenationLayer(1,2,'Name','cat')
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(1)]);

criticNet = connectLayers(criticNet,'state','cat/in1');
criticNet = connectLayers(criticNet,'action','cat/in2');

critic = rlQValueFunction(criticNet, obsInfo, actInfo);

% TD3 OPTIONS(v imp)
agent = rlTD3Agent(actor, critic, ...
    rlTD3AgentOptions( ...
        'SampleTime', p.dt, ...
        'TargetSmoothFactor', 5e-4, ...
        'DiscountFactor', 0.995, ...
        'PolicyUpdateFrequency', 2));


% Training
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', 400, ...
    'Verbose', true, ...
    'Plots','training-progress');

trainingStats = train(agent, env, trainOpts);
save trainedAgent agent
