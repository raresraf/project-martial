#include <bits/stdc++.h>
using namespace std;

int main(void) {
  map<char, int> m;
  string s;

  cin >> s;

  for (auto i = s.end() - 1; i >= s.begin(); --i) {
    if (*i != '+') {
      m[*i]++;
    }
  }

  s = "";

  for (auto i = m.begin(); i != m.end();) {
    s += i->first;

    if (!(i->second)) {
      ++i;
    } else {
      i->second--;

      if (!(i->second)) {
        ++i;
      }

      if (i != m.end()) {
        s += '+';
      }
    }
  }

  cout << s;
  return 0;
}
