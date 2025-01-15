// sumanth2303
// codeforces - 112A
// problem link - https://codeforces.com/problemset/problem/112/A
// ACCEPTED -  18-11-2020
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  string s1, s2;
  cin >> s1 >> s2;

  for (int i = 0; i < s1.size(); i++) {

    if (tolower(s1[i]) != tolower(s2[i])) {
      int a = tolower(s1[i]);
      int b = tolower(s2[i]);
      a < b ? cout << -1 << endl : cout << 1 << endl;
      return 0;
    }
  }
  cout << 0 << endl;
  return 0;
}