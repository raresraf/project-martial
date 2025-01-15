#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s;
  char c;
  int count = 1;
  cin >> s;
  for (int i = 1; i < s.size(); i++) {

    if (s[i] == s[i - 1])
      count++;
    else
      count = 1;
    if (count == 7) {
      cout << "YES" << endl;
      break;
    }
    if (i == s.size() - 1) {
      cout << "NO" << endl;
    }
  }

  return 0;
}
