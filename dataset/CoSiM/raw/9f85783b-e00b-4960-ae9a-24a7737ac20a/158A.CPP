#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k, a[100], cnt = 0;
  scanf("%d %d", &n, &k);
  for (int i = 1; i <= n; i++) {
    scanf("%d", &a[i]);
  }
  int sav = a[k];
  for (int i = 1; i <= n; i++) {
    if (a[i] >= sav && a[i])
      cnt++;
  }
  printf("%d", cnt);
  return 0;
}
