
#include <iostream>

using namespace std;

int main() {
  int k, w;
  long long n, sum = 0;
  cin >> k >> n >> w;

  for (int i = 0, j = k; i < w; i++, j += k)
    sum += j;
  if (sum > n)
    cout << sum - n;
  else
    cout << 0;

  return 0;
}
