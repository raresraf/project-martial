#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main() {
  int n;
  char array[105];
  cin >> n;

  while (n) {
    cin >> array;
    if (strlen(array) > 10) {
      cout << array[0] << (strlen(array) - 2) << array[strlen(array) - 1]
           << endl;
    } else
      cout << array << endl;

    n--;
  }

  return 0;
}
