#include <bits/stdc++.h>
using namespace std;

int main() {
  string s1, s2;
  cin >> s1;
  cin >> s2;
  for (int i = 0; i < s1.length(); i++) {
    s1[i] = towlower(s1[i]);
    s2[i] = towlower(s2[i]);
  }
  if (s1 == s2) {
    cout << 0;
  }
  if (s1 > s2) {
    cout << 1;
  } else if (s1 < s2) {
    cout << -1;
  }

  return 0;
}
