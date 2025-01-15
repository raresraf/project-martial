// https://codeforces.com/problemset/problem/263/A
#include <cmath>
#include <iostream>

using namespace std;
int main() {
  int x, row, col;
  for (row = 1; row <= 5; ++row) {
    for (col = 1; col <= 5; ++col) {
      cin >> x;
      if (x == 1) {
        cout << abs(row - 3) + abs(col - 3) << endl;
      }
    }
  }
}