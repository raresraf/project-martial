#include <iostream>

using namespace std;

bool A(int num) {
  if (num == 2)
    return false;

  return num % 2 == 0;
}

void main() {
  int count;
  cin >> count;

  cout << (A(count) ? "YES" : "NO") << endl;
}