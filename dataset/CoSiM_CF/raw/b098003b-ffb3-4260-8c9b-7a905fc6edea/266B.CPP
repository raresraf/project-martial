/*
 *Link: https://codeforces.com/problemset/problem/266/B
 */

#include <iostream>
#include <string>
using namespace std;

string update(string a) {
  for (int i = 0; i < a.length() - 1; i++) {
    if (a[i] == 'B' && a[i + 1] == 'G') {
      a[i] = 'G';
      a[i + 1] = 'B';
      i++;
    }
  }
  return a;
}

int main() {
  unsigned int n, t;
  string s;
  cin >> n >> t >> s;
  for (int i = 0; i < t; i++) {
    s = update(s);
  }
  cout << s << endl;
}