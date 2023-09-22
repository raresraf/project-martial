py_library(
  name = "comments",
  deps = [
    "//grammars/go:grammar_go_lexers", 
    ":comments_helpers",
    ":comments_config",
  ],
  visibility = ["//visibility:public"],
  srcs = ["comments.py"],
)

py_library(
  name = "comments_helpers",
  visibility = ["//visibility:public"],
    deps = [
      ":comments_config",
  ],
  srcs = ["comments_helpers.py"],
)


py_library(
  name = "comments_config",
  visibility = ["//visibility:public"],
  srcs = ["comments_config.py"],
)


py_library(
  name = "comments_api",
  deps = [":comments", ":comments_config"],
  visibility = ["//visibility:public"],
  srcs = ["comments_api.py"],
)