/** 71A - Way Too Long Words
 * http://codeforces.com/problemset/problem/71/A
 * Category: Implementation
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s.size() <= 10)
      cout << s << "\n";
    else
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << "\n";
  }
  return 0;
}
