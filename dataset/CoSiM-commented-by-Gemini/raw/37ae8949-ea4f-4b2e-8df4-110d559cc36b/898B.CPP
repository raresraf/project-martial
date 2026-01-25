#include <iostream>
using namespace std;
long long n, a, b;
int main() {
  cin >> n;
  cin >> a >> b;
  if (n % a == 0)
    return cout << "YES\n" << n / a << " " << 0, 0;
  if (n % b == 0)
    return cout << "YES\n" << 0 << " " << n / b, 0;
  for (int i = 1; i < 10000000; i++) {
    long long x = n - (a * i);
    if (x % b == 0 && x >= 0)
      return cout << "YES\n" << i << " " << (x == 0 ? 0 : x / b), 0;
  }
  for (int i = 1; i < 10000000; i++) {
    long long x = n - (b * i);
    if (x % a == 0 && x >= 0)
      return cout << "YES\n" << (x == 0 ? 0 : x / b) << " " << i, 0;
  }
  cout << "NO";
  return 0;
}
