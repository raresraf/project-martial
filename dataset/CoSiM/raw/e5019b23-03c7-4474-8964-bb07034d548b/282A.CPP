#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, ct = 0;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    if (s.find("++") != string::npos)
      ct++;
    else if (s.find("--") != string::npos)
      ct--;
  }
  cout << ct << endl;
  return 0;
}
