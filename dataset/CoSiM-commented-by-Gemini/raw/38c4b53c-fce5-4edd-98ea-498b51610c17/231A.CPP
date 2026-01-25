#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, total = 0, a, b, c;

  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a;
    cin >> b;
    cin >> c;
    if (a + b + c >= 2) {
      total++;
    }
  }
  cout << total;
}
