#include <cstdio>

int main() {
  freopen("input", "rt", stdin);
  freopen("output", "wt", stdout);
  int n;
  scanf("%d", &n);
  int min = int(1e9), max = int(-1e9);
  for (int i = 0; i < 7; ++i) {
    int pos = i, ans = 0;
    for (int j = 0; j < n; ++j) {
      if (5 == pos || 6 == pos) {
        ++ans;
      }
      pos = (6 == pos ? 0 : pos + 1);
    }
    if (ans < min) {
      min = ans;
    }
    if (max < ans) {
      max = ans;
    }
  }
  printf("%d %d\n", min, max);
  return 0;
}
