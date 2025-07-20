/*
 *Link: https://codeforces.com/problemset/problem/112/A
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  string a, b;
  cin >> a >> b;
  for (int i = 0; i < a.length(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
    if (a[i] > b[i]) {
      cout << 1 << endl;
      return 0;
    } else if (a[i] < b[i]) {
      cout << -1 << endl;
      return 0;
    }
  }
  cout << 0 << endl;
  return 0;
}