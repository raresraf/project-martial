#include <bits/stdc++.h>
using namespace std;

int main() {
  map<char, int> m;
  string s;
  cin >> s;
  int result = 0;
  for (int i = 0; i < s.length(); i++) {
    m[s[i]]++;
  }
  for (map<char, int>::iterator it = m.begin(); it != m.end(); ++it) {
    result += 1;
  }
  if (result % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }
  return 0;
}
