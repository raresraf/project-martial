#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

using namespace std;

string lower(string s) {
  for (int i = 0, j = s.length(); i < j; i++) {
    s[i] = tolower(s[i]);
  }
  return s;
}

int main() {
  // freopen("input.txt", "r", stdin);

  string m;
  cin >> m;
  string n;
  cin >> n;
  m = lower(m);
  n = lower(n);

  if (m < n)
    cout << -1;
  if (m > n)
    cout << 1;
  if (m == n)
    cout << 0;
}
