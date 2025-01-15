#include <bits/stdc++.h>

using namespace std;

int main() {
  cin.tie(0)->sync_with_stdio(0);
  cout.tie(0);

  int T;
  cin >> T;
  while (T--) {
    int n, ans = 0;
    cin >> n;
    unordered_map<int, int> vec;
    for (int i = 0; i < n; i++) {
      int a;
      cin >> a;
      vec[a]++;
    }
    for (int i = 0; i <= 101; i++) {
      if (!vec[i]) {
        ans += i;
        break;
      }
      vec[i]--;
    }
    for (int i = 0; i <= 101; i++) {
      if (!vec[i]) {
        ans += i;
        break;
      }
    }
    cout << ans << "\n";
  }
}
