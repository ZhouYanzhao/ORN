import re


class FunctionBackend(object):

    def __init__(self, lib):
        self.backends = dict()
        self.parse_lib(lib)
        self.current_backend = None

    def __getattr__(self, name):
        func = self.backends[self.current_backend].get(name)
        if func is None:
            raise NotImplementedError(name)
        return func

    def set_type(self, input_type):
        if input_type != self.current_backend:
            if not input_type in self.backends.keys():
                raise NotImplementedError("{} is not supported".format(input_type))
            self.current_backend = input_type

    def parse_lib(self, lib):
        for func in dir(lib):
            if func.startswith('_'):
                continue
            match_obj = re.match(r"(\w+)_(Float|Double)_(.+)", func)
            if match_obj:
                if match_obj.group(1).startswith("cu"):
                    backend = "torch.cuda.{}Tensor".format(match_obj.group(2))
                else:
                    backend = "torch.{}Tensor".format(match_obj.group(2))
                func_name = match_obj.group(3)
                if backend not in self.backends.keys():
                    self.backends[backend] = dict()
                self.backends[backend][func_name] = getattr(lib, func)