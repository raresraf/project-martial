#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;

int n;
string s;

string convert(int cur) {
  string res;
  while (cur > 0) {
    res = (char)('0' + cur % 10) + res;
    cur /= 10;
  }
  return res;
}

int main() {
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> s;
    if (s.length() <= 10)
      cout << s << endl;
    else {
      string t = s[0] + convert(s.length() - 2) + s[s.length() - 1];
      cout << t << endl;
    }
  }

  return 0;
}
