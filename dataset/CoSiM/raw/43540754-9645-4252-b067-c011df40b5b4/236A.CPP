// author @Nishant

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  set<char> un;
  for (char x : s) {
    un.insert(x);
  }
  if (un.size() % 2) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
  return 0;
}