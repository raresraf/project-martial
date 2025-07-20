#include <iostream>

using namespace std;

int main() {
  int n, pet, vas, ton, number(0);

  cin >> n;

  if (!(1 <= n && n <= 1000)) {
    return 0;
  } else {
    ;
  }

  while (n--) {
    cin >> pet >> vas >> ton;
    for (int x : {pet, vas, ton}) {
      switch (x) {
      case 0:
      case 1:
        break;
      default:
        return 0;
      }
    }
    if (pet + vas + ton >= 2) {
      number += 1;
    }
  }
  cout << number << endl;
  return 0;
}