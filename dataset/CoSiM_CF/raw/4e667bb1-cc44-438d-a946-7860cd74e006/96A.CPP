#include <bits/stdc++.h>
using namespace std;

int main() {
  std::ios_base::sync_with_stdio(false);
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  string s;
  cin >> s;
  if ((s.find("1111111") == s.npos) && (s.find("0000000") == s.npos)) {
    cout << "NO\n";
  } else {
    cout << "YES\n";
  }
  return 0;
}
