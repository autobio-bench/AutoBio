import sympy as sp

class dK(sp.Function):
    _expr: sp.Expr
    _deep: sp.Expr
    _grad: list[sp.Expr]

    def _eval_derivative(self, s):
        try:
            index = self.args.index(s)
            return self.fdiff(index + 1)
        except ValueError:
            return 0

    def fdiff(self, argindex):
        if self._grad[argindex - 1] is None:
            grad = sp.diff(self._expr, self.args[argindex - 1])
            self._grad[argindex - 1] = grad
        return self._grad[argindex - 1]

    def doit(self, **hints):
        if hints.get('deep', True):
            if self._deep is None:
                self._deep = self._expr.doit(**hints)
            return self._deep
        else:
            return self._expr
    
    @staticmethod
    def derive(name: str, expr: sp.Expr, *args: sp.Symbol) -> 'dK':
        d = {
            "_expr": expr,
            "_deep": None,
            "_grad": [None] * len(args)
        }
        dC = type(name, (dK,), d)
        return dC(*args)

class K(sp.Function):
    _expr: sp.Expr
    _deep: sp.Expr
    _grad: list[sp.Expr]

    @classmethod
    def eval(cls, *args):
        for sym in args:
            # only symbols are allowed
            assert sym.is_Symbol

    def _eval_derivative(self, s):
        try:
            index = self.args.index(s)
            return self.fdiff(index + 1)
        except ValueError:
            return 0

    def fdiff(self, i):
        i -= 1
        if self._grad[i] is None:
            name = f"dq{i}{self.__class__.__name__}"
            expr = sp.diff(self._expr, self.args[i])
            self._grad[i] = dK.derive(name, expr, *self.args)
        return self._grad[i]

    def doit(self, **hints):
        if hints.get('deep', True):
            if self._deep is None:
                self._deep = self._expr.doit(**hints)
            return self._deep
        else:
            return self._expr
    
    @staticmethod
    def derive(name: str, expr: sp.Expr, *args: sp.Symbol) -> 'K':
        d = {
            "_expr": expr,
            "_deep": None,
            "_grad": [None] * len(args)
        }
        C = type(name, (K,), d)
        return C(*args)
