#include <bits/stdc++.h>
#include <stdio.h>
#include <string.h>
#include <vector>
using namespace std;
int main() {
  int i, l, c = 0;
  string s;
  cin >> s;
  l = s.size();
  std::sort(s.begin(), s.end());
  for (i = 0; i < l; i++) {
    if (s[i] != s[i + 1])
      c++;
  }
  if (c % 2 == 0) {
    cout << "CHAT WITH HER!" << endl;
  } else
    cout << "IGNORE HIM!" << endl;
  return 0;
}
