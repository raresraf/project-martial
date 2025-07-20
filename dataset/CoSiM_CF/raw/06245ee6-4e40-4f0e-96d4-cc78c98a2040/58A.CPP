#include <bits/stdc++.h>
using namespace std;
vector<char> vv;
int main() {
  string s, s2 = "hello";
  cin >> s;
  int j = 0, len = s.length();
  for (int i = 0; i < len; i++) {
    if (s[i] == s2[j]) {
      j++;
      vv.push_back(s[i]);
    }
  }
  if (vv.size() == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
