#include <bits/stdc++.h>
using namespace std;

int main(void) {
  string s;
  cin >> s;
  int ans = 0;

  for (int i = 0; i < s.length(); ++i) {
    if (s[i] == '7' || s[i] == '4') {
      ans++;
    }
  }
  cout << ((ans == 4 || ans == 7) ? "YES" : "NO") << '\n';
  return 0;
}
