#include <iostream>

using namespace std;

int main() {
  int k, n, w, price;

  cin >> k >> n >> w;

  price = k;
  for (int bananas = 2; bananas <= w; bananas++)
    price += bananas * k;

  if (price > n)
    cout << price - n;
  else
    cout << 0;

  return 0;
}
