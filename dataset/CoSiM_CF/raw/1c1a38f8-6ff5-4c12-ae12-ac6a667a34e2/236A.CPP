/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/236/A
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  cin >> a;
  sort(a.begin(), a.end());
  auto r = unique(a.begin(), a.end());
  a = string(a.begin(), r);
  a = a.length() % 2 == 0 ? "CHAT WITH HER!" : "IGNORE HIM!";
  cout << a << endl;
}
