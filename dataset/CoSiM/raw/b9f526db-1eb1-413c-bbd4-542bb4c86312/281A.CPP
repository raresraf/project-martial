// By archit-dev, contest: Codeforces Round #172 (Div. 2), problem: (A) Word
// Capitalization, Accepted, #, Copy
#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <string>
using namespace std;

int main() {
  string s;
  cin >> s;
  if (s[0] >= 97) {
    s[0] = s[0] - 32;
  }
  cout << s << "\n";
  return 0;
}
