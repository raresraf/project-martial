// Problem Link https://codeforces.com/problemset/problem/263/A

#include <iostream>

using namespace std;

int main() {
  int x, y;
  int num;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      cin >> num;
      if (num == 1) {
        x = i;
        y = j;
      }
    }
  }

  int moves = abs(x - 2) + abs(y - 2);
  cout << moves << "\n";
}