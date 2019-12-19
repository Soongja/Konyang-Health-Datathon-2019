def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, _BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return group_decay, group_no_decay


# build optimizer
if 'no_bias_decay' in config.train and config.train.no_bias_decay:
    if 'encoder_lr_ratio' in config.train:
        encoder_lr_ratio = config.train.encoder_lr_ratio
        group_decay_encoder, group_no_decay_encoder = group_weight(model.encoder)
        group_decay_decoder, group_no_decay_decoder = group_weight(model.decoder)
        base_lr = config.optimizer.params.lr
        params = [{'params': group_decay_decoder},
                  {'params': group_no_decay_decoder, 'weight_decay': 0.0},
                  {'params': group_decay_encoder, 'lr': base_lr * encoder_lr_ratio},
                  {'params': group_no_decay_encoder, 'lr': base_lr * encoder_lr_ratio, 'weight_decay': 0.0}]
    else:
        group_decay, group_no_decay = group_weight(model)
        params = [{'params': group_decay},
                  {'params': group_no_decay, 'weight_decay': 0.0}]
elif 'encoder_lr_ratio' in config.train:
    denom = config.train.encoder_lr_ratio
    base_lr = config.optimizer.params.lr
    params = [{'params': model.decoder.parameters()},
              {'params': model.encoder.parameters(), 'lr': base_lr * encoder_lr_ratio}]
else:
    params = model.parameters()
optimizer = build_optimizer(config, params=params)
