#!/bin/Rscript

args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("At least one argument must be supplied")
} else {
  data_path = args[1]
}

library(htmltools)
library(conumee2)
library(sesame)
library(plotly)

message("Running: ", data_path)
message("Loading idats from: ", file.path(data_path, "idats/"))
sdfs <- openSesame(file.path(data_path, "idats/"), func=NULL, prep="QCDPB")

message("Converting sdfs object")
name <- names(searchIDATprefixes(file.path(data_path, "idats/")))

temp_list <- list()
temp_list[[name]] <- sdfs %>% as.data.frame
sdfs <- temp_list

message("CNVs calling")
reference_path <- "/ref_data/"

reference <- openSesame(reference_path, prep = "QCDPB", func = NULL)
reference <- CNV.load(do.call(cbind, lapply(reference, totalIntensities)))

sample <- CNV.load(totalIntensities(sdfs[[name]]), names=name)

data(exclude_regions)
data(detail_regions)

anno <- CNV.create_anno(array_type = c("450k", "EPIC", "EPICv2"), exclude_regions = exclude_regions, detail_regions = detail_regions)
cnvs <- CNV.fit(query = sample, ref = reference, anno)
cnvs <- CNV.bin(cnvs)
cnvs <- CNV.detail(cnvs)
cnvs <- CNV.segment(cnvs)

cnvs_plot <- CNV.plotly(cnvs)
cnvs_json <- plotly_json(cnvs_plot, jsonedit = FALSE)
writeLines(cnvs_json, "cnvs.json")

message("DONE")
