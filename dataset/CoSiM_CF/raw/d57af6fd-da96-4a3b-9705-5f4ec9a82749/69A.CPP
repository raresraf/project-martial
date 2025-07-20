#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int n, x, y, z, xcount = 0, ycount = 0, zcount = 0;

  scanf("%d", &n);

  while (n--) {
    scanf("%d %d %d", &x, &y, &z);

    xcount += x;
    ycount += y;
    zcount += z;
  }

  if ((xcount == 0) && (ycount == 0) && (zcount == 0)) {
    printf("YES\n");
  } else {
    printf("NO\n");
  }

  return 0;
}
