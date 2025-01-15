#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, x = 0;
  cin >> n;
  string op;
  for (int i = 0; i < n; i++) {
    cin >> op;
    if (op[1] == '+')
      x++;
    else if (op[1] == '-')
      x--;
  }
  cout << x;
  return 0;
}