#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b, c, sum, cnt = 0;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    sum = 0;
    cin >> a >> b >> c;
    sum = a + b + c;
    if (sum >= 2)
      cnt++;
  }
  cout << cnt << endl;
  return 0;
}
