#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int t, n, x, k, sum, toBeAdded, i;
  long long int a[200000];
  int prevSign;
  scanf("%lld", &t);
  while (t--) {
    sum = 0;
    scanf("%lld", &n);
    scanf("%lld", &a[0]);
    prevSign = a[0] > 0 ? 1 : 0;
    toBeAdded = a[0];
    for (i = 1; i < n; i++) {
      scanf("%lld", &a[i]);
      if (a[i] < 0 && prevSign == 1) {
        prevSign = 0;
        sum += toBeAdded;
        toBeAdded = a[i];
      } else if (a[i] > 0 && prevSign == 0) {
        prevSign = 1;
        sum += toBeAdded;
        toBeAdded = a[i];
      } else {
        if (a[i] > toBeAdded)
          toBeAdded = a[i];
      }
    }
    sum += toBeAdded;
    printf("%lld\n", sum);
  }
  return 0;
}
