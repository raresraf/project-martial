#include <bits/stdc++.h>

using namespace std;

int main(void) {
  int n;
  cin >> n;
  while (n--) {
    string s;
    cin >> s;
    if (s.length() > 10) {
      std::string l = std::to_string(s.length() - 2);
      cout << (s[0] + l + s[s.length() - 1]) << endl;
    } else {
      cout << s << endl;
    }
  }
}