#include <iostream>
using namespace std;
int main() {
  long long int a, b, c, k;
  cin >> a >> b >> c;
  k = (a * (c * (c + 1)) / 2);
  if (k <= b)
    cout << "0\n";
  else
    cout << k - b;
  return 0;
}
