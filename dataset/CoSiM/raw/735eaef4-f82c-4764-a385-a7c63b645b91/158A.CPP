#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main() {
  int x;
  int k;
  int array[105];
  cin >> x;
  cin >> k;
  int cnt = 0;

  for (int i = 0; i < x; i++) {
    cin >> array[i];
  }
  int m = array[k - 1];

  for (int i = 0; i < x; i++) {
    if (array[i] > 0)
      if (array[i] >= m)
        cnt++;
  }
  cout << cnt;
  return 0;
}
