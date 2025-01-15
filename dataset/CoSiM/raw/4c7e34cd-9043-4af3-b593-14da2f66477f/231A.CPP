#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n, co = 0, i, a, b, c;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%d %d %d", &a, &b, &c);
    if (a == b && b == c && c == 1) {
      co++;
    } else if (a == b && b == 1) {
      co++;
    } else if (b == c && c == 1) {
      co++;
    } else if (a == c && c == 1) {
      co++;
    }
  }
  printf("%d\n", co);
  return 0;
}
