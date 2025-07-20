/*
Link: https://codeforces.com/problemset/problem/266/B
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  string s, a;
  int n, t, size, i, j;
  cin >> n >> t;
  cin >> s;
  a = s;
  for (i = 0; i < t; i++) {
    for (j = 1; j < n; j++) {
      if (a[j - 1] == 'B' && a[j] == 'G') {
        a[j - 1] = 'G';
        a[j] = 'B';
        j++;
      }
    }
  }
  cout << a;
}
