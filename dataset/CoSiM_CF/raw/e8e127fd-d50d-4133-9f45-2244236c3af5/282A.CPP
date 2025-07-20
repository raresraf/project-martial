#include <cstring>
#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  char op[6];
  int n, x = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> op;
    if ((strcmp(op, "++X")) == 0)
      ++x;
    if ((strcmp(op, "X++")) == 0)
      x++;
    if ((strcmp(op, "--X")) == 0)
      --x;
    if ((strcmp(op, "X--")) == 0)
      x--;
    // cout<<x;
  }
  cout << x;
  return 0;
}
