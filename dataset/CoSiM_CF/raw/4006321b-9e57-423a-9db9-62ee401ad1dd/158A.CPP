#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  int n, k;

  cin >> n >> k;

  int counter = 0;
  int a[100];
  for (int i = 0; i < n; i++) {
    cin >> a[i];
  }
  for (int i = 0; i < n; i++) {
    if (a[i] >= a[k - 1] && a[i] > 0)
      counter++;
  }

  cout << counter;
  return 0;
}
