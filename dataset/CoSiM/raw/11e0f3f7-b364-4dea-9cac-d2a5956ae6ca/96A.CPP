#include <bits/stdc++.h>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  getline(cin, s);
  int count = 1;

  int i = 0;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == s[i + 1]) {
      count++;
    } else if (s[i] != s[i + 1]) {
      if (count < 7) {
        count = 1;
      } else {
        continue;
      }
    }
  }

  if (count >= 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
