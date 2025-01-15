/*
Link: https://codeforces.com/problemset/problem/339/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  string s;
  int c1 = 0, c2 = 0, c3 = 0, cp = 0, i = 0, count = 0;
  cin >> s;
  while (s[i] != '\0') {
    if (s[i] == '1')
      c1++;
    else if (s[i] == '2')
      c2++;
    else if (s[i] == '3')
      c3++;
    else
      cp++;
    i++;
  }
  i = 0;
  while (count < c1) {
    s[i] = '1';
    if (cp != 0) {
      i++;
      s[i] = '+';
      cp--;
    }
    i++;
    count++;
  }
  count = 0;
  while (count < c2) {
    s[i] = '2';
    if (cp != 0) {
      i++;
      s[i] = '+';
      cp--;
    }
    i++;
    count++;
  }
  count = 0;
  while (count < c3) {
    s[i] = '3';
    if (cp != 0) {
      i++;
      s[i] = '+';
      cp--;
    }
    i++;
    count++;
  }
  cout << s;
  return 0;
}