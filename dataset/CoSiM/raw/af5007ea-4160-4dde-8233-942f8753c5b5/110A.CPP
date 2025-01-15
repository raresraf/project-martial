#include <iostream>

using std::cin;
using std::cout;
using std::endl;

int main() {
  long long n;
  cin >> n;
  int luckyDigits = 0;
  while (n != 0) {
    if (n % 10 == 4 || n % 10 == 7) {
      luckyDigits++;
    }
    n /= 10;
  }

  if (luckyDigits == 0) {
    cout << "NO";
    return 0;
  }
  while (luckyDigits != 0) {
    if (luckyDigits % 10 != 4 && luckyDigits % 10 != 7) {
      cout << "NO";
      return 0;
    }
    luckyDigits /= 10;
  }

  cout << "YES";
  return 0;
}
