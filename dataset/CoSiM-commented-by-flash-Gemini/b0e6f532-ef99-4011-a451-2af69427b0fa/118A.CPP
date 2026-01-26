#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, f = "aoyeuiAOYEUI";
  cin >> s;
  int i = -1;
  while (++i < s.size()) {
    if (f.find(s[i]) != string::npos)
      continue;
    else
      cout << "." << (char)(s[i] > 'Z' ? s[i] : s[i] + ('a' - 'A'));
  }
}
