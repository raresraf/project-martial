#include <algorithm>
#include <bits/stdc++.h>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string s, t, revers;
  while (cin >> s >> t) {
    reverse(s.begin(), s.end());
    if (s == t) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}
