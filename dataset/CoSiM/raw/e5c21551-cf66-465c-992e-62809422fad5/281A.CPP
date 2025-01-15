#include <bits/stdc++.h>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  string s;
  cin >> s;
  char c = s[0];
  c = toupper(c);
  s[0] = c;
  cout << s;
  return 0;
}