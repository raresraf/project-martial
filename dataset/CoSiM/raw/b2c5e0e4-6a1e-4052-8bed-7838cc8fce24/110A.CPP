/* Petya loves lucky numbers. We all know that lucky numbers are the positive
integers whose decimal representations contain only the lucky digits 4 and 7.
For example, numbers 47, 744, 4 are lucky and 5, 17, 467 are not.

Unfortunately, not all numbers are lucky. Petya calls a number nearly lucky if
the number of lucky digits in it is a lucky number. He wonders whether number n
is a nearly lucky number.

Input
The only line contains an integer n (1   n   10^18).

Please do not use the %lld specificator to read or write 64-bit numbers in  ++.
It is preferred to use the cin, cout streams or the %I64d specificator.

Output
Print on the single line "YES" if n is a nearly lucky number. Otherwise, print
"NO" (without ' the quotes). */

#include <iostream>

using namespace std;

int main() {
  long long n;
  cin >> n;
  int ct = 0;
  while (n > 0) {
    int digit = n % 10;
    if (digit == 4 || digit == 7)
      ct++;
    n = n / 10;
  }
  if (ct == 4 || ct == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}