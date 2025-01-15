#include <bits/stdc++.h>
using namespace std;

int main(void) {
  map<char, int> m;
  string s;

  cin >> s;

  for (auto i = s.begin(); i != s.end(); ++i) {
    m[*i]++;
  }

  if (m.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else {
    cout << "IGNORE HIM!";
  }

  return 0;
}
