options(Ncpus = 8)

install.packages(c("devtools", "arrow", "randomForest", "jsonlite", "htmlwidgets", "ggplot2", "plotly", "BiocManager"))

BiocManager::install("preprocessCore", configure.args = c(preprocessCore = "--disable-threading"), force=TRUE, update=TRUE, type = "source")
BiocManager::install(c("minfi", "Rsamtools", "RnBeads", "illuminaio", "nullranges"))
devtools::install_github("hovestadtlab/conumee2", subdir = "conumee2")

BiocManager::install("sesame")
