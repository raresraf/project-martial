py_binary(
  name = "comments_driver",
  deps = [
    "//modules:comments",
    "//modules:comments_api",
    "//modules:comments_config",
  ],
  srcs = ["comments_driver.py"],
)

py_binary(
  name = "comments_dump",
  deps = [
    "//modules:comments",
    "//modules:comments_api",
    "//modules:comments_config",
  ],
  srcs = ["comments_dump.py"],
)

py_binary(
  name = "rcomplexity_driver",
  deps = [
    "//modules:rcomplexity",
    "//modules:rcomplexity_api",
  ],
  srcs = ["rcomplexity_driver.py"],
)

py_binary(
  name = "rcomplexity_gradient_descent",
  deps = [
    ":rcomplexity_driver",
    "//modules:rcomplexity",
    "//modules:rcomplexity_api",
  ],
  srcs = ["rcomplexity_gradient_descent.py"],
)
