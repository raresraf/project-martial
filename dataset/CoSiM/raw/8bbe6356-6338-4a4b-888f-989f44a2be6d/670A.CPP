#include <iostream>
using namespace std;

int main() {
  int n, mi = 0, ma = 0;
  scanf("%d", &n);
  ma += (n / 7) * 2 + min(2, n % 7);
  n -= 5;
  if (n > 5)
    mi += 2 * (n / 7);
  if (n > 0)
    mi += min(2, n % 7);
  cout << mi << " " << ma << endl;
}
