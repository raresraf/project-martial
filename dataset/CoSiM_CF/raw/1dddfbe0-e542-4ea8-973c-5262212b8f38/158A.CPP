#include <cstdio>
#include <iostream>

using namespace std;

int main() {
  int n, k, a[51], sum = 0, i;

  scanf("%d %d", &n, &k);

  for (i = 0; i < n; ++i) {
    scanf("%d", &a[i]);
    if (i < k && a[i] > 0)
      ++sum;
    else if (a[i] >= a[k - 1] && a[i] > 0)
      ++sum;
  }
  printf("%d\n", sum);

  return 0;
}
