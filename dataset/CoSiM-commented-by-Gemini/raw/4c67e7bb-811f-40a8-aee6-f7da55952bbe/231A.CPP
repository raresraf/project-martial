#include <iostream>
using namespace std;
int s[1111];
int main() {
  int n, a, b, c, sum = 0;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> a >> b >> c;
    s[i] = a + b + c;
    if (s[i] > 1)
      ++sum;
  }
  cout << sum;
}
