#include <iostream>

using std::cin;
using std::cout;

int main(void) {
  int weight;

  cin >> weight;
  cout << ((weight % 2 == 0 && weight > 2) ? "YES" : "NO");

  return 0;
}