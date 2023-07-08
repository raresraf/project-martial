py_library(
  name = "comments",
  deps = [
    "//grammars/go:grammar_go_lexers", 
    ":comments_helpers",
  ],
  visibility = ["//visibility:public"],
  srcs = ["comments.py"],
)

py_library(
  name = "comments_helpers",
  visibility = ["//visibility:public"],
  srcs = ["comments_helpers.py"],
)


py_library(
  name = "comments_api",
  deps = [":comments"],
  visibility = ["//visibility:public"],
  srcs = ["comments_api.py"],
)