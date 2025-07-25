load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "eventhandlers.go",
        "factory.go",
        "scheduler.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/v1/pod:go_default_library",
        "//pkg/controller/volume/scheduling:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/scheduler/algorithmprovider:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/apis/config/scheme:go_default_library",
        "//pkg/scheduler/apis/config/validation:go_default_library",
        "//pkg/scheduler/core:go_default_library",
        "//pkg/scheduler/framework/plugins:go_default_library",
        "//pkg/scheduler/framework/plugins/defaultbinder:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/framework/plugins/queuesort:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//pkg/scheduler/internal/cache/debugger:go_default_library",
        "//pkg/scheduler/internal/queue:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//pkg/scheduler/profile:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/policy/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/config/v1alpha2:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "eventhandlers_test.go",
        "factory_test.go",
        "scheduler_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/api/testing:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/controller/volume/scheduling:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/apis/config/scheme:go_default_library",
        "//pkg/scheduler/core:go_default_library",
        "//pkg/scheduler/framework/plugins:go_default_library",
        "//pkg/scheduler/framework/plugins/defaultbinder:go_default_library",
        "//pkg/scheduler/framework/plugins/interpodaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodelabel:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeports:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/framework/plugins/queuesort:go_default_library",
        "//pkg/scheduler/framework/plugins/serviceaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/volumebinding:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//pkg/scheduler/internal/cache/fake:go_default_library",
        "//pkg/scheduler/internal/queue:go_default_library",
        "//pkg/scheduler/profile:go_default_library",
        "//pkg/scheduler/testing:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/events/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/tools/events:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/extender/v1:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//pkg/scheduler/algorithmprovider:all-srcs",
        "//pkg/scheduler/apis/config:all-srcs",
        "//pkg/scheduler/core:all-srcs",
        "//pkg/scheduler/framework:all-srcs",
        "//pkg/scheduler/internal/cache:all-srcs",
        "//pkg/scheduler/internal/heap:all-srcs",
        "//pkg/scheduler/internal/parallelize:all-srcs",
        "//pkg/scheduler/internal/queue:all-srcs",
        "//pkg/scheduler/metrics:all-srcs",
        "//pkg/scheduler/profile:all-srcs",
        "//pkg/scheduler/testing:all-srcs",
        "//pkg/scheduler/util:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
