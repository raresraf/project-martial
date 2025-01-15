#include <bits/stdc++.h>
using namespace std;
int main() {
  int l, count = 0;
  string s;
  cin >> s;
  l = s.length();
  sort(s.begin(), s.end());

  for (int i = 1; i < l; i++) {
    if (s[i] != s[i - 1]) {
      count++;
    }
  }

  if ((count + 1) % 2 == 0) {
    cout << "CHAT WITH HER!\n" << endl;
  } else {
    cout << "IGNORE HIM!\n" << endl;
  }
  return 0;
}