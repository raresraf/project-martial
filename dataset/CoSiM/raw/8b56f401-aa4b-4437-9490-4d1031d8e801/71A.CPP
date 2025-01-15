// By archit-dev, contest: Codeforces Beta Round #65 (Div. 2), problem: (A) Way
// Too Long Words, Accepted, #, Copy
#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <string>
#define ll long long int
#define ull unsigned long long int
using namespace std;

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int t;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    if (s.size() <= 10) {
      cout << s << "\n";
    } else {
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << "\n";
    }
  }
  return 0;
}
