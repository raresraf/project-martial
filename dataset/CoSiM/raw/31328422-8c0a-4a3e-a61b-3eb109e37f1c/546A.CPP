#include <iostream>

using namespace std;

int main() {
  int k{}, n{}, w{}, price{};
  cin >> k >> n >> w;

  for (int i = 1; i <= w; i++)
    price += (i * k);
  if (price > n)
    cout << price - n;
  else
    cout << 0;

  return 0;
}