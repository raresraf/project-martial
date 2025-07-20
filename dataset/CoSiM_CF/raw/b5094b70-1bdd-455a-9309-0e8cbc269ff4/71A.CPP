#include <bits/stdc++.h>
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  _ int n;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    if (s.length() <= 10) {
      cout << s << "\n";
    } else {
      cout << s[0] << s.length() - 2 << s[s.length() - 1] << "\n";
    }
  }
  return 0;
}