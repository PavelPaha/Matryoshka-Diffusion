def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module
