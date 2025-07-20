#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int ans = 0;
  int a = -1;
  int x, y;
  for (int i = 0; i < n; i++) {
    cin >> x >> y;
    ans = (ans - x + y);
    a = max(a, ans);
  }
  cout << a << endl;
  return 0;
}
