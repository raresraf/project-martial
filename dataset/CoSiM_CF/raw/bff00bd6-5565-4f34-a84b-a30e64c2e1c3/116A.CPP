#include <bits/stdc++.h>
using namespace std;
int main() {
  int i, n, a, b, cnt = 0, sum = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> a >> b;
    cnt = (cnt + b) - a;
    if (cnt >= sum) {
      sum = cnt;
    }
  }
  cout << sum;
  return 0;
}
