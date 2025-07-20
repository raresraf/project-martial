#include <iostream>
using namespace std;

int main() {

  int n;
  char c1;
  char c2;
  int x = 0;
  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> c1 >> c2 >> c2;

    if (c1 == '+' || c2 == '+') {
      x++;
    } else {
      x--;
    }
  }

  cout << x << endl;

  return 0;
}
