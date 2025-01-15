#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, a, b, sum = 0, count = 0, max = 0, x, y, sum2 = 0;
  cin >> n;

  int l = n;

  for (int i = 0; i < l; i++) {
    if (i == 0) {
      cin >> a >> b;
      sum = a + b;
      if (sum > max) {
        max = sum;
      }
    } else {

      cin >> x >> y;
      sum2 = sum - x;
      count = sum2 + y;
      sum = count;
      if (count > max) {
        max = count;
      }
    }
  }

  cout << max << endl;
}
