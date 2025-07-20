// Link to Problem: https://codeforces.com/problemset/problem/71/A

#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    int si = s.length();
    if (si > 10) {
      cout << s[0] << si - 2 << s[si - 1] << '\n';
    } else
      cout << s << '\n';
  }

  return 0;
}