// 56680198	Practice:havoccboi	96A - 10	GNU C++17	Accepted
// 62 ms	16 KB	2019-07-07 20:54:01	2019-07-07 20:54:01

#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  int c = 1;
  bool flag = false;
  if ((int)s.size() < 7) {
    cout << "NO" << endl;
  } else {
    for (int i = 0; i < (int)s.size() - 1; i++) {
      if (s[i] == s[i + 1]) {
        c++;
      } else {
        c = 1;
      }

      if (c == 7) {
        flag = true;
        break;
      }
    }
    if (flag == true)
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
}
