// By archit-dev, contest: Codeforces Round #163 (Div. 2), problem: (A) Stones
// on the Table, Accepted, #, Copy
#include <algorithm>
#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  int n, steps = 0, k = 0;
  cin >> n;
  string s;
  cin >> s;
  char last = s[k];
  for (int i = 1; i < s.size(); i++) {
    if (last == s[i]) {
      steps += 1;
    }
    k += 1;
    last = s[k];
  }
  cout << steps;
  return 0;
}
