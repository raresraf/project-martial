#include <bits/stdc++.h>
using namespace std;

int main() {
  string s, sf = ".";
  cin >> s;
  char av[6] = {'A', 'O', 'Y', 'E', 'U', 'I'};

  for (int i = 0; i < s.size(); ++i) {
    char token = 'n';
    for (int j = 0; j < 6; ++j) {
      if (s[i] == av[j] || s[i] == (char)(av[j] + 32)) {
        token = 'y';
        break;
      }
    }
    if (token == 'y')
      continue;

    if (s[i] < 97)
      s[i] += 32;
    sf += s[i];
    if (i < s.size())
      sf += '.';
  }
  sf.erase(sf.size() - 1, sf.size());
  cout << sf << endl;

  return 0;
}

// 'A' = 65
// 'a' = 97
// dif = 32
// '.' = 46
// codeforces.com/problemset/problem/118/A