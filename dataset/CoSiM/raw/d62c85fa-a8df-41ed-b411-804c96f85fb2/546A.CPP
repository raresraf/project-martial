#include <iostream>
using namespace std;

int main() {
  long k, n, w; // k == cost of first banana || n == dollars soldier || w ==
                // bananas he wants
  cin >> k >> n >> w;
  long price = 0;
  for (int i = 1; i <= w; i++) {
    price += (i * k);
  }
  if (n >= price) {
    cout << "0" << endl;
  } else {
    cout << price - n << endl;
  }
  return 0;
}
