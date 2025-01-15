#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int r;
  string s;

  cin >> r >> s;

  r = 0;

  for (auto i = s.begin(); i < s.end() - 1; ++i) {
    if (*i == *(i + 1)) {
      ++r;
    }
  }

  cout << r;
  return 0;
}
