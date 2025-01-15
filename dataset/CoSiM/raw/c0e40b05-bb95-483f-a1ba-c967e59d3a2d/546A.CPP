#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main() {

  int dollar;
  int wallet;
  int bananas;
  int inx = 1;
  int total = 0;
  cin >> dollar >> wallet >> bananas;

  while (bananas) {
    total += (dollar * inx);

    inx++;
    bananas--;
  }
  if (wallet >= total)
    cout << "0";
  else
    cout << (total - wallet);

  return 0;
}
