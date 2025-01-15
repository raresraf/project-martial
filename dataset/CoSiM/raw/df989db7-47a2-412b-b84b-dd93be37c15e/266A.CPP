#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;

int main() {
  int n, m, a, b, c = 0, x, y, z;

  // freopen("A.txt","r",stdin);
  scanf("%d%d%d%d", &n, &m, &a, &b);

  if (n % m > 0) {
    x = (n / m) * b + b;
    y = (n / m) * b + a * (n % m);
    z = n * a;
    c = min(x, min(y, z));
  } else {
    x = n * a;
    y = (n / m) * b;
    c = min(x, y);
  }
  printf("%d\n", c);

  return 0;
}
