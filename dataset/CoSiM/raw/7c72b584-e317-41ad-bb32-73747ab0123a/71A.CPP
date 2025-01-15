#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string s;
    cin >> s;
    if (s.length() < 11)
      cout << s << endl;
    else {
      string r = s[0] + to_string(s.length() - 2) + s[s.length() - 1];
      cout << r << endl;
    }
  }
}