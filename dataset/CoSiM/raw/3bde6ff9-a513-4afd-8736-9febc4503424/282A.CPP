/*
Link: https://codeforces.com/problemset/problem/282/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  int n, x = 0, i, k;
  string op;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> op;
    if (op[0] == '+' || op[1] == '+')
      x++;
    else
      x--;
  }
  cout << x;
  return 0;
}