#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main() {
  int sum1 = 0;
  int sum2 = 0;
  int flag = 0;
  char array1[105];
  char array2[105];
  cin >> array1;
  cin >> array2;
  for (int i = 0; i < strlen(array1); i++) {
    if (isupper(array1[i])) {
      array1[i] = tolower(array1[i]);
    }
    if (isupper(array2[i])) {
      array2[i] = tolower(array2[i]);
    }

    if (array1[i] == array2[i]) {
      flag = 0;
    } else if (array1[i] > array2[i]) {
      flag = 1;
      break;
    } else if (array1[i] < array2[i]) {
      flag = 2;
      break;
    }
  }

  switch (flag) {
  case 0: {
    cout << "0";
    break;
  }
  case 1: {
    cout << "1";
    break;
  }
  case 2: {
    cout << "-1";
    break;
  }
  }

  return 0;
}
