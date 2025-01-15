#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, arr[1001][3], i, j, cnt = 0, ck = 0;

  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++)
      scanf("%d", &arr[i][j]);
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < 3; j++) {
      if (arr[i][j])
        cnt++;
    }
    if (cnt >= 2)
      ck++;
    cnt = 0;
  }
  printf("%d\n", ck);

  return 0;
}
