#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  char hi[] = {'h', 'e', 'l', 'l', 'o'};

  int j = 0;
  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == hi[j])
      j++;
    if (j >= 5)
      break;
  }

  (j >= 5) ? cout << "YES\n" : cout << "NO\n";

  return 0;
}

// https://codeforces.com/problemset/problem/58/A
