#include <bits/stdc++.h>
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  _ int n;
  cin >> n;
  int ans = 0;
  char x, last = '.';

  for (int i = 0; i < n; i++) {
    cin >> x;
    if (x == last) {
      ans++;
    } else
      last = x;
  }
  cout << ans;
  return 0;
}