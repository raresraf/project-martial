#include <algorithm>
#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  string first;
  cin >> s;

  first = s[0];
  s = s.replace(0, 1, "");
  transform(first.begin(), first.end(), first.begin(), ::toupper);
  cout << first << s;

  return 0;
}
