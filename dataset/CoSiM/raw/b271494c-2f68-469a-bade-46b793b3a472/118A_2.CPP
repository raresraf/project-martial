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
  cin >> array;
  for (int i = 0; i < strlen(array); i++) {
    if (isupper(array[i])) {
      array[i] = tolower(array[i]);
    }

    if (array[i] != 'a' && array[i] != 'e' && array[i] != 'o' &&
        array[i] != 'i' && array[i] != 'u' && array[i] != 'y') {
      cout << "." << array[i];
    }
  }

  return 0;
}
