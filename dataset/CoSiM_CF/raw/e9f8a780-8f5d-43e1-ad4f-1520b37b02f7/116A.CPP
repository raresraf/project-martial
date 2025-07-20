#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b, sum, mx;
  while (cin >> n) {
    sum = 0;
    mx = 0;
    while (n--) {
      cin >> a >> b;
      sum = (sum - a) + b;
      if (sum > mx)
        mx = sum;
    }

    cout << mx << endl;
  }

  return 0;
}
