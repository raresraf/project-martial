#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  for (int i = 0; i < n; ++i) {
    string s;
    cin >> s;
    if (s.size() > 10) {
      int a = s.size() - 2;
      printf("%c%d%c\n", s[0], a, s[s.size() - 1]);
    } else
      cout << s << endl;
  }

  return 0;
}

// codeforces.com/problemset/problem/71/A