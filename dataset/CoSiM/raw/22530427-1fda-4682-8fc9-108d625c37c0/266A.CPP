#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  char a[55];
  int n;
  char prev;
  int ans = 0;
  scanf("%d %s", &n, a);
  prev = a[0];
  for (int i = 1; i < n; ++i) {
    if (a[i] == prev) {
      ++ans;
    } else {
      prev = a[i];
    }
  }
  printf("%d\n", ans);
  return 0;
}