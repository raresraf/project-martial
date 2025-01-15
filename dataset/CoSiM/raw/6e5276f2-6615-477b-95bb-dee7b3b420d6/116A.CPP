#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  scanf("%d", &n);

  int np = 0, mc;
  int a, b;
  while (n--) {
    scanf("%d %d", &a, &b);
    np -= a;
    np += b;
    mc = max(np, mc);
  }

  printf("%d", mc);

  return 0;
}
