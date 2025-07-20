#include <bits/stdc++.h>
using namespace std;
string hello = "hello";
string res = "NO";
string s;
int main() {
  int k = 0;
  cin >> s;
  for (int i = 0; i < (int)s.size(); i++) {
    if (hello[k] == s[i]) {
      k++;
    }
    if (k == (int)hello.size()) {
      res = "YES";
      break;
    }
  }
  cout << res << '\n';
}
