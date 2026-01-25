/// Problem Name: Chat room
/// Problem Link: https://codeforces.com/problemset/problem/58/A

#include <bits/stdc++.h>

using namespace std;

int main() {
  int i = 0, j = 0;
  string s;
  string h = "hello";
  cin >> s;
  int len = s.size();
  while (i < len && j < 5) {
    if (s[i] == h[j]) {
      i++;
      j++;
    } else
      i++;
  }
  if (j == 5)
    puts("YES");
  else
    puts("NO");
  return 0;
}
