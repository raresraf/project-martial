#include <iostream>
using namespace std;

int main() {
  int n, x = 0;
  char c[3];

  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> c[0] >> c[1] >> c[2];
    if (c[0] == '+' || c[1] == '+') {
      x++;
    } else if (c[0] == '-' || c[1] == '-') {
      x--;
    }
  }
  cout << x << endl;
}
