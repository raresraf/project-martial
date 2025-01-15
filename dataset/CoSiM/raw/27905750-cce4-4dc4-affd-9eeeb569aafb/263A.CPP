#include <bits/stdc++.h>
int main() {
  int x = 0, y = 6, x2 = 0, y2 = 0, l;
  for (int i = 0; i < 25; ++i) {
    scanf("%i", &l);
    if (i % 5 == 0 && i != 0) {
      x = 0;
    }
    if (x == 0) {
      y--;
    }
    x++;
    if (l == 1) {
      x2 = x;
      y2 = y;
    }
  }
  printf("%i\n", abs(3 - x2) + abs(3 - y2));
  return 0;
}
