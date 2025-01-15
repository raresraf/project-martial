// By archit-dev, contest: Codeforces Beta Round #85 (Div. 2 Only), problem: (A)
// Petya and Strings, Accepted, #, Copy
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  int flag = 0;
  string s1, s2;
  cin >> s1 >> s2;
  for (int i = 0; i < s1.size(); i++) {
    s1[i] = toupper(s1[i]);
  }
  for (int i = 0; i < s2.size(); i++) {
    s2[i] = toupper(s2[i]);
  }
  for (int i = 0; i < s1.size(); i++) {
    if (s1[i] > s2[i]) {
      flag = 1;
      break;
    } else if (s1[i] < s2[i]) {
      flag = -1;
      break;
    } else {
      continue;
    }
  }
  cout << flag << "\n";
  return 0;
}
