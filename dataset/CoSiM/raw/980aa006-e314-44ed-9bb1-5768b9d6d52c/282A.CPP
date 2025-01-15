/*
 * Problem description:
 *
 * https://codeforces.com/problemset/problem/282/A
 *
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);

  int16_t x{0};
  string e;

  cin.ignore();

  while (cin >> e) {
    if (e == "X++" || e == "++X")
      x++;
    else if (e == "X--" || e == "--X")
      x--;
  }
  cout << x << '\n';
}
