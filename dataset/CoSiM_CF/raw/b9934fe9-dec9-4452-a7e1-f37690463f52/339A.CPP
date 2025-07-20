#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  string s;
  cin >> s;
  int one = 0, two = 0, three = 0;
  if (s.length() == 1) {
    cout << s << endl;
  } else {
    for (int i = 0; i < s.length(); i += 2) {
      if (s[i] == '1') {
        one += 2;
      }
      if (s[i] == '2') {
        two += 2;
      }
      if (s[i] == '3') {
        three += 2;
      }
    }
    for (int i = 0; i < one; i += 2) {
      s[i] = '1';
    }
    for (int i = one; i < one + two; i += 2) {
      s[i] = '2';
    }
    for (int i = one + two; i < one + two + three; i += 2) {
      s[i] = '3';
    }
    cout << s << endl;
  }

  return 0;
}
