#include <iostream>

using namespace std;

int main() {
  int k = 0, n = 0, w = 0;
  cin >> k >> n >> w;
  int borrow = 0;
  int total = 0;
  for (int i = 1; i < (w + 1); i++) {
    total = total + (i * k);
  }

  borrow = total - n;
  if (borrow < 0) {
    borrow = 0;
  }
  cout << borrow << endl;
}
