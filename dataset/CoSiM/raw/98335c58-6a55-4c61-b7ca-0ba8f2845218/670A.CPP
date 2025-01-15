// Codeorces 670A- Holidays

#include <iostream>
#include <math.h>
using namespace std;
int main() {
  long long n, minimum, maximum, weeks, days;
  cin >> n;

  weeks = n / 7;
  days = n % 7;
  minimum = weeks * 2;
  if (days == 6)
    minimum += 1;
  maximum = weeks * 2;
  if (days >= 2)
    maximum += 2;
  else if (days == 1)
    maximum += 1;

  cout << minimum << " " << maximum;
}
