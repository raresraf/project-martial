#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    (s.length() <= 10)
        ? cout << s << "\n"
        : cout << s[0] << (s.length() - 2) << s[s.length() - 1] << "\n";
  }
}