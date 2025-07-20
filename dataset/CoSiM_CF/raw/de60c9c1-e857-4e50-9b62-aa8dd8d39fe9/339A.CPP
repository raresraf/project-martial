#include <bits/stdc++.h>

using namespace std;

int main() {
  string s;
  cin >> s;
  string r(s.length(), '+');
  int num1 = 0, num2 = 0, num3 = 0;
  for (int i = 0; i < s.length(); i += 2) {
    switch (s[i]) {
    case '1':
      num1++;
      break;
    case '2':
      num2++;
      break;
    case '3':
      num3++;
      break;
    default:
      continue;
    }
  }
  for (int i = 0; i < r.length(); i += 2) {
    if (num1) {
      r[i] = '1';
      num1--;
    } else if (num2) {
      r[i] = '2';
      num2--;
    } else if (num3) {
      r[i] = '3';
      num3--;
    }
  }
  cout << r;
}