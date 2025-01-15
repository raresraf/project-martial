#include "bits/stdc++.h"
using namespace std;
int main() {
  string s;
  long long ans = 0;
  cin >> s;
  map<char, int> m;
  for (int i = 0; i < s.length(); i++) {
    m[s[i]]++;
  }
  for (int i = 0; i < s.length(); i++)
    ans = ans + m[s[i]];
  cout << ans;
  return 0;
}