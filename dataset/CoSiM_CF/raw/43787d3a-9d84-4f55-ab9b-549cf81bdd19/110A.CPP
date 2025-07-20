#include <iostream>

using namespace std;

int main() {
  uint64_t n, lucky_digits_total = 0;
  unsigned int d;

  cin >> n;

  while (n >= 10) {
    d = n % 10;
    n /= 10;
    if (d == 4 || d == 7)
      lucky_digits_total++;
  }
  if (n == 4 || n == 7)
    lucky_digits_total++;

  while (lucky_digits_total >= 10) {
    d = lucky_digits_total % 10;
    lucky_digits_total /= 10;
    if (d != 4 && d != 7) {
      cout << "NO" << endl;
      exit(0);
    }
  }
  if (lucky_digits_total != 4 && lucky_digits_total != 7)
    cout << "NO" << endl;
  else
    cout << "YES" << endl;
}
