#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int nCoins, sum = 0, counter = 0, nsum = 0, temp = 0;
  scanf("%i", &nCoins);
  int cValues[nCoins];
  for (int i = 0; i < nCoins; i++) {
    scanf("%i", &cValues[i]);
    sum += cValues[i];
  }
  for (int i = 0; i < nCoins; i++) {
    for (int j = 0; j < nCoins - 1; j++) {
      if (cValues[j] > cValues[j + 1]) {
        temp = cValues[j];
        cValues[j] = cValues[j + 1];
        cValues[j + 1] = temp;
      }
    }
  }

  for (int i = nCoins - 1; i >= 0; i--) {
    nsum += cValues[i];
    counter++;

    if (nsum > sum - nsum) {
      break;
    }
  }
  printf("%i", counter);
  return 0;
}
