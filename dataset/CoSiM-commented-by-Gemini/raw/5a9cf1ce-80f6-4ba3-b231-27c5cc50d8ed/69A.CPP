#include <iostream>

using namespace std;

int main() {
  int n;
  int x, y, z;
  int sx = 0, sy = 0, sz = 0;
  scanf("%d", &n);
  while (n--) {
    scanf("%d %d %d", &x, &y, &z);
    sx += x;
    sy += y;
    sz += z;
  }
  if (sx == 0 && sy == 0 && sz == 0) {
    printf("YES");
  } else {
    printf("NO");
  }
}