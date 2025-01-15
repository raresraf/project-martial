#include <bits/stdc++.h>
using namespace std;

int main() {
  bool danger = false;
  string s;
  int o = 0, z = 0;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    if (i == 0) {
      if (s[0] == '0')
        z++;
      else
        o++;
    } else {
      if (i < s.length() - 1) {
        if (s[i] == s[i + 1]) {
          if (s[i] == '0')
            z++;
          else
            o++;
        } else {
          if (s[i] == '0') {
            z++;
            if (z >= 7) {
              danger = true;
              break;
            } else {
              z = 0;
            }
          } else {
            o++;
            if (o >= 7) {
              danger = true;
              break;
            } else {
              o = 0;
            }
          }
        }
      } else {
        if (s[i] == '0') {
          z++;
          if (z >= 7) {
            danger = true;
            break;
          }
        } else {
          o++;
          if (o >= 7) {
            danger = true;
            break;
          }
        }
      }
    }
  }
  if (danger)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
