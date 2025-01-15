#include <cstring>
#include <iostream>
using namespace std;
main() {
  int ops, x = 0;
  string compute;
  cin >> ops;
  while (ops) {
    cin >> compute;
    if (compute == "X++" || compute == "++X") {
      x++;
    } else {
      x--;
    }
    ops--;
  }
  cout << x;
}
