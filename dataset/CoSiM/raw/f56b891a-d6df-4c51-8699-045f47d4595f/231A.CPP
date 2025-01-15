#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, cnt = 0, a, b, c;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    if ((a + b + c) >= 2) {
      cnt++;
    }
  }
  cout << cnt;
}
