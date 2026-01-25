/*
 * Problem description
 *
 * https://codeforces.com/problemset/problem/112/A
 *
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int a;
  string i1, i2;
  cin >> i1 >> i2;
  transform(i1.begin(), i1.end(), i1.begin(), ::tolower);
  transform(i2.begin(), i2.end(), i2.begin(), ::tolower);
  a = i1 > i2 ? 1 : (i1 < i2 ? -1 : 0);
  cout << a;
  return 0;
}
