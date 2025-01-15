// By archit-dev, contest: Codeforces Beta Round #84 (Div. 2 Only), problem: (A)
// Nearly Lucky Number, Accepted, #, Copy

#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <string>
using namespace std;

int main() {
  long long int lucky = 0;
  string s;
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      lucky += 1;
    }
  }
  bool luck = false;
  while (lucky > 0) {
    int rem = lucky % 10;
    if (rem == 4 || rem == 7) {
      luck = true;
      lucky /= 10;
    } else {
      luck = false;
      break;
    }
  }
  if (luck) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
  return 0;
}
