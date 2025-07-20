/*
 *Link: https://codeforces.com/problemset/problem/282/A
 */

#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, x = 0;
  string statement;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> statement;
    if (statement[1] == '+')
      x++;
    else
      x--;
  }
  cout << x;
}