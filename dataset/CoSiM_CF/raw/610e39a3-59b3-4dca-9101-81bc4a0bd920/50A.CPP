#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  long long x, y, total;
  cin >> x;
  cin >> y;
  total = x * y;
  if (total % 2 == 0) {
    cout << total / 2;
  } else {
    cout << (total - 1) / 2;
  }

  return 0;
}
