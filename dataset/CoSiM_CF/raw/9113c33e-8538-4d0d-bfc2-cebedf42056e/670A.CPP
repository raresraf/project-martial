#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  scanf("%d", &n);
  int mx = (n / 7) * 2, mn = (n / 7) * 2;

  if (n % 7 == 6)
    mn++;
  if (n % 7 == 1)
    mx++;
  else if (n % 7 >= 2)
    mx += 2;

  printf("%d %d\n", mn, mx);
  return 0;
}
