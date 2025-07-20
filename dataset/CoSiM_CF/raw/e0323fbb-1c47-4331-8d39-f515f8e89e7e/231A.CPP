#include <algorithm>
#include <cstdio>

using namespace std;

int main() {
  int n, i, ans = 0;
  scanf("%d", &n);
  int x[n][3];
  for (i = 0; i < n; i++) {
    scanf("%d %d %d", &x[i][0], &x[i][1], &x[i][2]);
    sort(x[i], x[i] + 3);
    if (x[i][1] && x[i][2]) {
      ans++;
    }
  }
  printf("%d", ans);
  return 0;
}