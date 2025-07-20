#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k, a[101], count = 0, i, null = 0;
  scanf("%d %d\n", &n, &k);
  for (i = 1; i <= n; i++) {
    scanf("%d", &a[i]);
    null += a[i];
  }
  if (null != 0) {

    for (i = 1; i <= n; i++) {
      if (a[i] >= a[k] && a[k] != 0) {
        count = count + 1;
      } else if (a[i] > a[k] && a[k] == 0) {
        if (a[i] > 0) {
          count += 1;
        }
      }
    }
  } else {
    count = 0;
  }
  cout << count << endl;
}
