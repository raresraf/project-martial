#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, m;
  scanf("%d %d", &n, &m);
  int i = m / 2;
  i *= n;
  int x = m % 2, y = n / 2;
  if (x > 0 && y > 0)
    i += x * y;
  printf("%d\n", i);
  return 0;
}
