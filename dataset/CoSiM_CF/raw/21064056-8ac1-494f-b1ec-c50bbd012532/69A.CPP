#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, a = 0, b = 0, c = 0, x, y, z;
  scanf("%d", &n);
  for (int i = 1; i <= n; i++) {
    cin >> x >> y >> z;
    a += x;
    b += y;
    c += z;
  }
  if (a == 0 && b == 0 && c == 0)
    printf("YES");
  else
    printf("NO");
}
