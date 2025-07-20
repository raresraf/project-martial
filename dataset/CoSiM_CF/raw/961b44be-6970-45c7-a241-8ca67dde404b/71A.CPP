#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  string s[100];
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }
  for (int i = 0; i < n; i++) {
    if (s[i].length() > 10)
      cout << s[i][0] << s[i].length() - 2 << s[i][s[i].length() - 1] << endl;
    else
      cout << s[i] << endl;
  }
  return 0;
}
