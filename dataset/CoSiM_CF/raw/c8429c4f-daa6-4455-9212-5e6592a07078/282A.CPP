#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, x = 0;
  string operation;

  cin >> n;
  if (1 <= n && n <= 150) {
    while (n--) {
      cin >> operation;

      if (operation[0] == 'X') {
        if (operation[1] == '+' && operation[2] == '+') {
          x++;
        } else if (operation[1] == '-' && operation[2] == '-') {
          x--;
        }

      } else if (operation[0] == '+') {
        if (operation[1] == '+' && operation[2] == 'X') {
          x++;
        }

      } else if (operation[0] == '-') {
        if (operation[1] == '-' && operation[2] == 'X') {
          x--;
        }
      }
    }
  }
  cout << x;
  return 0;
}
