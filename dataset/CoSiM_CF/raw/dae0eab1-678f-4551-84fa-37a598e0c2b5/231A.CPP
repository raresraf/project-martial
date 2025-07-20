#include <bits/stdc++.h>
using namespace std;
int main() {
  int t, a, b, c, sum = 0;
  for (scanf("%d", &t); t--;) {
    scanf("%d %d %d", &a, &b, &c);
    if (a + b + c > 1)
      sum++;
  }
  printf("%d\n", sum);
  return 0;
}
