reg = [

    ("SVR", SVR()),
    ("Kernel Ridge RBF", Regression_With_Custom_Kernel(KernelRidge(), RBF())),
    ("Kernel Ridge Cauchy", Regression_With_Custom_Kernel(KernelRidge(), Cauchy())),
    ("Kernel Ridge Tanimoto", Regression_With_Custom_Kernel(KernelRidge(), Tanimoto()))
]