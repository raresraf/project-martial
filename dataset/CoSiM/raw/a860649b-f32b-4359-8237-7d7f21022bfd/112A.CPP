#include <bits/stdc++.h>
using namespace std;

int main(void) {
  string s, t;

  cin >> s >> t;

  for (int i = 0; i < s.length(); ++i) {
    if (s[i] >= 97 && s[i] <= 122) {
      if (t[i] >= 97 && t[i] <= 122) {
        if (s[i] < t[i]) {
          cout << "-1" << endl;
          return 0;
        }
        if (s[i] > t[i]) {
          cout << "1" << endl;
          return 0;
        }
      } else {
        if (s[i] < t[i] + 32) {
          cout << "-1" << endl;
          return 0;
        }
        if (s[i] > t[i] + 32) {
          cout << "1" << endl;
          return 0;
        }
      }
    } else {
      if (t[i] >= 65 && t[i] <= 90) {
        if (s[i] < t[i]) {
          cout << "-1" << endl;
          return 0;
        }
        if (s[i] > t[i]) {
          cout << "1" << endl;
          return 0;
        }
      } else {
        if (s[i] < t[i] - 32) {
          cout << "-1" << endl;
          return 0;
        }
        if (s[i] > t[i] - 32) {
          cout << "1" << endl;
          return 0;
        }
      }
    }
  }

  cout << "0" << endl;
  return 0;
}
