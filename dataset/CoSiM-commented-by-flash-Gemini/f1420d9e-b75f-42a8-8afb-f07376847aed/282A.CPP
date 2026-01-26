#include <bits/stdc++.h>
using namespace std;

int main() {
  int t;
  cin >> t;
  int ans = 0;

  for (int i = 0; i < t; i++) {
    // int ans=0;
    string s;
    cin >> s;
    if (s[0] == '+') {
      ++ans;
    }
    if (s[0] == '-') {
      --ans;
    }
    if (s[2] == '+') {
      ans++;
    }
    if (s[2] == '-') {
      ans--;
    }
    // cout << ans << endl;
  }
  cout << ans << endl;

  return 0;
}
