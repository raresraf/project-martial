#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int a, b, sum = 0, max = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    sum -= a;
    sum += b;
    if (sum > max)
      max = sum;
  }
  cout << max;
  return 0;
}
