#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  long long k, n, w, money = 0, ans;
  cin >> k;
  cin >> n;
  cin >> w;
  for (int i = 0; i <= w; i++) {
    money = money + i * k;
  }
  if (money > n) {
    ans = money - n;
    cout << ans;
  } else {
    cout << "0";
  }
  return 0;
}
