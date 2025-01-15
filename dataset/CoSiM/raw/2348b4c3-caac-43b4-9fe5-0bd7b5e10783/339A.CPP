/* Date: 10.01.16
Problem: 339A - Helpful Maths
*/
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>

int compare(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

using namespace std;

int main() {

  int i, n, m, t = 0, k, w = 0;
  string c;
  int arr[100];

  cin >> c;

  for (i = 0; i < c.length(); i += 2) {
    arr[t] = c[i] - '0';
    t++;
  }

  qsort(arr, t, sizeof(int), compare);

  t = 0;
  for (i = 0; i < c.length(); i += 2) {
    if (i == c.length() - 1) {
      cout << arr[t] << endl;
      break;
    }
    cout << arr[t] << "+";
    t++;
  }

  return 0;
}
