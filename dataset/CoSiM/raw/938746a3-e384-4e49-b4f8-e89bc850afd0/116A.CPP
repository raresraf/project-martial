#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n, a, b, c(0), m(0);
  scanf("%d", &n);
  while (n--) {
    scanf("%d%d", &a, &b);
    c -= a;
    c += b;
    if (c > m)
      m = c;
  }
  printf("%d\n", m);
  return 0;
}
