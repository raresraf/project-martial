#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main() {
  char array[105];
  int m;
  cin >> array;
  int n = strlen(array);
  for (int i = 0; i < strlen(array); i++) {
    m = 0;
    for (int j = i + 1; j < strlen(array); j++) {
      if (array[i] == array[j] && (m == 0)) {
        n--;
        m++;
      }
    }
  }

  if (n % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
}
