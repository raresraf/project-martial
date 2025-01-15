#include <algorithm>
#include <cmath>
#include <complex>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
using namespace std;
#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define REV(i, b, a) for (int i = (a); i >= (b); i--)
#define mp make_pair
#define pb push_back

int main(int argc, char *argv[]) {

  string s;
  short _1, _0;
  _1 = _0 = 0;

  cin >> s;

  FOR(i, 0, s.size()) {
    if (_0 == 6) {
      cout << "YES";
      return 0;
    } else if (_1 == 6) {
      cout << "YES";
      return 0;
    }
    if (s[i] == '0') {
      if (s[i] == s[i + 1] && s[i] == '0')
        _0++;
      else
        _0 = 0;
    } else {
      if (s[i] == s[i + 1] && s[i] == '1')
        _1++;
      else
        _1 = 0;
    }
  }

  cout << "NO";

  return 0;
}
