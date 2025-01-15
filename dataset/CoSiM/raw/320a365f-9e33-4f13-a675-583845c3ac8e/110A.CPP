// author @Nishant

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  int count = 0;
  for (int i = 0; i < s.size(); i++) {
    if (char(s[i]) - 49 == 3 || char(s[i]) - 49 == 6) {
      count++;
    }
  }
  if (count == 4 || count == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}