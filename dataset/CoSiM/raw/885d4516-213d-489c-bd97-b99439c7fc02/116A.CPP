/*
 *Link: https://codeforces.com/problemset/problem/116/A
 */

#include <iostream>
using namespace std;

int main() {
  int n, a, b, max = 0, pass = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    pass -= a;
    pass += b;
    if (max < pass)
      max = pass;
  }
  cout << max << endl;
}