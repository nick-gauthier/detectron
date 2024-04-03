#' @importFrom reticulate py_install virtualenv_exists virtualenv_remove
install_detectron <-
  function(...,
           envname = "r-detectron",
           new_env = identical(envname, "r-detectron")) {

    if(new_env && reticulate::virtualenv_exists(envname)) reticulate::virtualenv_remove(envname)

    package_list <- c("pyyaml==5.1",
                      "torch",
                      "torchvision",
                      "torchaudio",
                      "opencv-python",
                      "sahi")
    reticulate::py_install(packages = package_list, envname = envname, ...)
    reticulate::py_install('git+https://github.com/facebookresearch/detectron2.git', envname = envname, ...)
}

.onLoad <- function(...) {
  reticulate::use_virtualenv("r-detectron", required = FALSE)
}
