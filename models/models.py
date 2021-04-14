
def create_PRSNet(opt):
    from .PRSNetJittor import PRSNet, Inference
    if opt.isTrain:
        model = PRSNet()
    else:
        model = Inference()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))
    return model