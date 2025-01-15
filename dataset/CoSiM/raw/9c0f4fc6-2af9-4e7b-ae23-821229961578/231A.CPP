#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int i = 0, Z = 0;
  int a, b, c;

  while (i < n) {
    cin >> a >> b >> c;
    if (a + b + c >= 2) {
      Z++;
    }

    i++;
  }

  cout << Z;
}
