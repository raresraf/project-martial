#include <bits/stdc++.h>

using namespace std;

int main() {
#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  int n;
  cin >> n;
  string s;
  int x = 0;

  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s[1] == '+')
      x = x + 1;
    else
      x = x - 1;
  }

  cout << x;

  return 0;
}