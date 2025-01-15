#include <algorithm>
#include <iostream>
#include <stdio.h>
using namespace std;
main() {
  int n, ans1, ans2, ans, i;
  cin >> n;
  int a[n];
  int sum = 0;
  for (i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }

  sort(a, a + n);

  for (i = n - 1; i >= 0; i--) {
    ans1 += a[i];
    ans2 = sum - ans1;

    if (ans1 > ans2) {
      ans = i;
      break;
    }
  }
  cout << ans << endl;
  return 0;
}
