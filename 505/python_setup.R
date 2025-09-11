#Setting Up Python

library(reticulate)
use_python("C:\\Users\\manam\\AppData\\Local\\Programs\\Python\\Python313\\python.exe", required = TRUE)
virtualenv_create("myenv", python = "C:\\Users\\manam\\AppData\\Local\\Programs\\Python\\Python313\\python.exe")
py_install(c("pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"), envname = "myenv")
use_virtualenv("myenv", required = TRUE)
