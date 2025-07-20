#include <bits/stdc++.h>
#define lli long long int
using namespace std;

int main() {
  int n;
  cin >> n;
  while (n--) {
    lli c, m, x;
    cin >> c >> m >> x;
    int l = 0, sum = 0;
    int r = min(c, m);
    while (l <= r) {
      int mid = l + (r - l) / 2;
      if (c + m + x - 2 * mid >= mid) {
        l = mid + 1;
        sum = mid;
      } else {
        r = mid - 1;
      }
    }
    cout << sum << endl;
  }

  return 0;
}
