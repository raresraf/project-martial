#include <bits/stdc++.h>
using namespace std;

int in[1005];
int out[1005];

int main() {
  int n, cmp = 0, ans = 0, f = 0;
  scanf("%d", &n);
  int i;
  for (i = 0; i < n; i++) {
    scanf("%d %d", &out[i], &in[i]);
    ans -= out[i];
    ans += in[i];
    if (cmp < ans) {
      f = ans;
      cmp = ans;
    }
  }
  printf("%d\n", f);
  return 0;
}
