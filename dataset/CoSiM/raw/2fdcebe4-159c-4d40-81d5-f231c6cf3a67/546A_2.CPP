#include <iostream>
using namespace std;

int main() {
  int k, w;
  long long n, c;
  cin >> k >> n >> w;
  c = k * w * (w + 1) / 2;
  if (c <= n)
    cout << 0;
  else
    cout << c - n;
  return 0;
}
