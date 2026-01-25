/*
 * Problem description
 *
 * https://codeforces.com/problemset/problem/118/A
 *
 */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string a;
  stringstream ss;
  cin >> a;
  char ac;
  char v[6] = {'a', 'o', 'y', 'e', 'u', 'i'};
  for_each(begin(a), end(a), [](char &c) { c = tolower(c); });
  for (int i = 0; a[i] != '\0'; i++) {
    ac = a[i];
    bool ia = find(begin(v), end(v), ac) != end(v);
    if (!ia)
      ss << '.' << ac;
  }
  ss >> a;
  cout << a << endl;
  ;
  return 0;
}
