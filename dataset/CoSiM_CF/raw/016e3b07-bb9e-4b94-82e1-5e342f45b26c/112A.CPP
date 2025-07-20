#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;
  getline(cin, s1);
  getline(cin, s2);
  transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
  transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
  int res = 0;
  for (int i = 0; i < s1.length(); i++) {
    if (s1[i] > s2[i])
      res = 1;
    else if (s1[i] < s2[i])
      res = -1;
  }
  cout << res;
  return 0;
}
