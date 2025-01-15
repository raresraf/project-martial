#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  if (s.find("1111111") != s.npos || s.find("0000000") != s.npos)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
