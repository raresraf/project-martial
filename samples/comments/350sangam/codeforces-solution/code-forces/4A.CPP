#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  if (n > 0 && n < 101) {
    if (n % 2 == 0 && n / 2 != 1) {
      cout << "YES";
    } else {
      cout << "NO";
    }
  }

  return 0;
}