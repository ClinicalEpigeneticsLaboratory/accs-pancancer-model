#!/bin/Rscript

args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("At least one argument must be supplied")
} else {
  data_path = args[1]
}

library(randomForest)
library(jsonlite)
library(sesame)
library(plotly)
library(arrow)

message("Running: ", data_path)
message("Loading idats from: ", file.path(data_path, "idats/"))
sdfs <- openSesame(file.path(data_path, "idats/"), func=NULL, prep="QCDPB")

message("Inferring sex and platform from idats")
sex <- inferSex(openSesame(sdfs))
platform <- sesameData_check_platform(probes = sdfs$Probe_ID)
options <- list("MALE" = "Male", "FEMALE" = "Female")

write(toJSON(list("Predicted_sex" = options[[sex]], "Predicted_platform" = platform)), "results.json")

message("Converting sdfs object")
name <- names(searchIDATprefixes(file.path(data_path, "idats/")))
temp_list <- list()
temp_list[[name]] <- sdfs %>% as.data.frame
sdfs <- temp_list

message("Converting to betas")
betas = do.call(cbind, BiocParallel::bplapply(sdfs, getBetas))
betas = betasCollapseToPfx(betas)
betas = betas %>% as.data.frame

message("Exporting beta-matrix")
betas$CpG <- rownames(betas)
betas <- betas[grep("cg", rownames(betas)),]

write_parquet(betas, "mynorm.parquet")
message("Preprocessing: DONE")
