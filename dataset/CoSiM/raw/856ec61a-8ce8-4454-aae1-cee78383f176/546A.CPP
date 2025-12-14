#include <iostream>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;

  int sum = 0;
  for (int i = 1; i <= w; i++) {
    sum = sum + i * k;
  }
  int res = sum - n;
  if (res > 0)
    cout << res;
  else
    cout << 0;
  return 0;
}