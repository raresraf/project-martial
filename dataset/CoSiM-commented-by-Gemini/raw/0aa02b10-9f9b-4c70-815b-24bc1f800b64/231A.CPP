#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, cn = 0;
  scanf("%d", &n);
  int i;
  for (i = 0; i < n; i++) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    if (a == 1 && b == 1)
      cn++;
    else if (b == 1 && c == 1)
      cn++;
    else if (c == 1 && a == 1)
      cn++;
  }
  printf("%d\n", cn);
  return 0;
}
