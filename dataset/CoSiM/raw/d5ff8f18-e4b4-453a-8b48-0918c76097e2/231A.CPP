#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b, c;
  int sum;
  cin >> n;
  int cnt = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    sum = a + b + c;
    if (sum >= 2) {
      cnt++;
    }
  }
  cout << cnt;
}
