#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;
int main() {
  char array[1005];
  cin >> array;
  array[0] = toupper(array[0]);
  cout << array;
  return 0;
}
