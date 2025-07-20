#include <iostream>

using namespace std;

int main() {
  int n; // # of operations
  cin >> n;

  int x = 0; // variable

  for (int i = 0; i < n; i++) {
    char a, b, c;
    cin >> a >> b >> c;

    if (b == '+')
      x++;
    else
      x--;
  }

  cout << x;

  return 0;
}
