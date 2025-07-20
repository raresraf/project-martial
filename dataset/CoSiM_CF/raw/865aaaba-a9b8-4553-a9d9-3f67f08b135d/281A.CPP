#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  if (s[0] < 123 and s[0] > 96) {
    s[0] = toupper(s[0]);
  }
  cout << s << endl;
  return 0;
}