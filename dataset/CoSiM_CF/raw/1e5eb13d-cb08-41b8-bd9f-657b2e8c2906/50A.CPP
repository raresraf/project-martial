#include <iostream>
using namespace std;
int main() {
  int m, n;
  cin >> m >> n;
  if (m * n % 2 == 0) {
    cout << m * n / 2;
  } else {
    int value;
    value = (m - 1) * n / 2 + (n - 1) / 2;
    cout << value;
  }
}
