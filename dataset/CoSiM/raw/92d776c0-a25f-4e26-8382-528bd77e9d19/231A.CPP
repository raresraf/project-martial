#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, n, i, j, cnt = 0;
  scanf("%d", &t);
  for (i = 1; i <= t; i++) {
    int sum = 0;
    for (j = 1; j <= 3; j++) {
      scanf("%d", &n);
      sum += n;
    }
    if (sum > 1)
      cnt++;
  }
  printf("%d\n", cnt);
  return 0;
}
