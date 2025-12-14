#include <bits/stdc++.h>
using namespace std;
int main() {
  string p, s;
  int n, i, val1, val2;
  cin >> p >> s;
  n = p.length();
  for (i = 0; i < n; i++) {
    if (p[i] >= 65 && p[i] <= 90)
      val1 = p[i] - 65;
    else
      val1 = p[i] - 97;
    if (s[i] >= 65 && s[i] <= 90)
      val2 = s[i] - 65;
    else
      val2 = s[i] - 97;
    if (val1 < val2) {
      cout << "-1";
      return 0;
    }
    if (val1 > val2) {
      cout << "1";
      return 0;
    }
  }
  cout << "0";
  return 0;
}
