#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  int f = s.find('h');
  if (f != string::npos) {
    f = s.find('e', f + 1);
    if (f != string::npos) {
      f = s.find('l', f + 1);
      if (f != string::npos) {
        f = s.find('l', f + 1);
        if (f != string::npos) {
          f = s.find('o', f + 1);
          if (f != string::npos) {
            cout << "YES" << endl;
            return 0;
          }
        }
      }
    }
  }
  cout << "NO" << endl;
  return 0;
}
