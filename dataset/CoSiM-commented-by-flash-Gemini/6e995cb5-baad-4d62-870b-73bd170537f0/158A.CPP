#include <algorithm>
#include <cstdio>
using namespace std;

int main() {
  int n, k, pass;
  scanf("%d %d", &n, &k);
  int x[n];
  int i;
  for (i = 0; i < n; i++) {
    scanf("%d", &x[i]);
  }
  pass = x[k - 1];
  int ans = 0;
  sort(x, x + n);
  for (i = 0; i < n; i++) {
    if (x[i] != 0 && x[i] >= pass)
      ans++;
  }
  printf("%d", ans);
  return 0;
}
