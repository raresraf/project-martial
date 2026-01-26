#include <bits/stdc++.h>
#define _                                                                      \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);

using namespace std;

int main() {
  _ string s;
  set<char> m;
  char x;
  while (cin >> x) {
    if (x == '\n')
      break;
    m.insert(x);
  }
  if (m.size() % 2 == 0) {
    cout << "CHAT WITH HER!\n";
  } else {
    cout << "IGNORE HIM!\n";
  }

  return 0;
}