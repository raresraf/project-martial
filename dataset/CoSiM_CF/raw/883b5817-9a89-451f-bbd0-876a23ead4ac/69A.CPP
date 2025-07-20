#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, x, y, z, ax = 0, ay = 0, az = 0;
  scanf("%d", &n);
  for (int i = 0; i < n; i++)
    scanf("%d %d %d", &x, &y, &z), ax += x, ay += y, az += z;
  if (!ax && !ay && !az)
    printf("YES");
  else
    printf("NO");
  return 0;
}