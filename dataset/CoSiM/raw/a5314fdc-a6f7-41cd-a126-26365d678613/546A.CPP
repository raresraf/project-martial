#include <iostream>

using namespace std;

int main() {
  long long k, d, n, b = 0;
  cin >> k >> d >> n;
  for (int i = 1; i <= n; i++)
    b += i * k;
  cout << (b - d > 0 ? b - d : 0);
  return 0;
}