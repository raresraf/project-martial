#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b, k = 0, res = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    res -= a;
    res += b;
    if (res > k) {
      k = res;
    }
  }
  cout << k << endl;
}
