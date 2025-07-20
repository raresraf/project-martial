#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  int count = 1, val = s[0], flag = 0;
  for (int i = 1; i < s.size(); i++) {
    if (s[i] != val) {
      val = s[i];
      if (count >= 7) {
        flag = 1;
      }
      count = 0;
    }
    count++;
  }
  if (flag || count >= 7)
    cout << "YES";
  else
    cout << "NO";
}
