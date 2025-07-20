#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  scanf("%d", &n);

  int x, y, z;
  int xt = 0, yt = 0, zt = 0;
  while (n--) {
    scanf("%d %d %d", &x, &y, &z);
    xt += x;
    yt += y;
    zt += z;
  }

  if (xt == 0 && yt == 0 && zt == 0)
    printf("YES");
  else
    printf("NO");

  return 0;
}
