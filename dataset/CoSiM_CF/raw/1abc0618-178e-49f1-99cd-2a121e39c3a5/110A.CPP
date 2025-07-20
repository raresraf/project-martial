#include <bits/stdc++.h>
using namespace std;
int main() {
  int a;
  int flag = 0;
  string s;
  cin >> s;
  int cnt = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      cnt++;
    }
  }
  int rem;
  if (cnt > 0) {
    while (cnt > 0) {
      rem = cnt % 10;
      cnt = cnt / 10;
      if (rem != 4 && rem != 7) {
        flag = 1;
        break;
      }
    }
    if (flag == 0) {
      cout << "YES";
    } else {
      cout << "NO";
    }
  } else {
    cout << "NO";
  }
}
