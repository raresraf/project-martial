#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, a[105], sum, temp, i;
  scanf("%d", &n);
  for (i = 0; i < n; i++)
    scanf("%d", &a[i]);
  sort(a, a + n, greater<int>());
  sum = temp = 0;
  for (i = 0; i < n; i++)
    sum += a[i];

  for (i = 0; i < n; i++) {
    temp += a[i];
    if (temp > sum / 2)
      break;
  }
  printf("%d\n", i + 1);
  return 0;
}
