#include <iostream>

using namespace std;

int main() {
  int n, a, b, N = 0, max = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    N = N + b;
    N = N - a;
    if (N >= max)
      max = N;
  }
  cout << max;
  return 0;
}