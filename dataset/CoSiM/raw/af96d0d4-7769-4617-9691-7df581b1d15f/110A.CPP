#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  string s;

  while (cin >> s) {
    int len = s.size(), cnt = 0;
    for (int i = 0; i < len; i++) {
      if (s[i] == '4' || s[i] == '7')
        cnt++;
    }
    if (cnt == 7 || cnt == 4)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }

  return 0;
}
