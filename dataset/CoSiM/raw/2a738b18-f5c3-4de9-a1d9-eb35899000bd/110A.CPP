#include <algorithm>
#include <iostream>
#include <set>

using namespace std;

int main() {

  long long int a;
  cin >> a;

  int ndigits = 0;
  while (a != 0) {
    if (a % 10 == 4 || a % 10 == 7)
      ndigits++;
    a /= 10;
  }

  if (ndigits == 4 || ndigits == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
