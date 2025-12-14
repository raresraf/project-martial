#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, a, b, total = 0, max = 0;

  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a;
    cin >> b;
    total = total - a + b;
    if (total > max) {
      max = total;
    }
  }
  cout << max;
}
