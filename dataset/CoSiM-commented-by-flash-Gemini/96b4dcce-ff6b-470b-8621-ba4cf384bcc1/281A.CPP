#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;
int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  string s;
  cin >> s;
  int x;
  x = int(s[0]);
  if (x >= 97) {
    s[0] = x - 32;
  }
  cout << s;
}