#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  if (s[0] >= 'a' && s[0] <= 'z')
    s[0] -= 'z' - 'Z';
  cout << s;
  return 0;
}
