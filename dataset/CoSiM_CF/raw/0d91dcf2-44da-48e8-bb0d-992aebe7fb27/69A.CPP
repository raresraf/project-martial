#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n, m, a, b;
  cin >> n >> m >> a >> b;

  int64_t res[3];
  res[0] = n * a, res[1] = ((n / m) * b);
  res[2] = res[1];

  int64_t temp = n % m;
  if (temp) {
    res[1] += b * temp;
    res[2] += a * temp;
  }
  sort(res, res + 3);
  cout << res[0];

  return 0;
}
