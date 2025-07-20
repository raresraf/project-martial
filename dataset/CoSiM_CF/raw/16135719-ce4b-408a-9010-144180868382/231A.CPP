#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n, dec, hit, miss, flag = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    hit = 0;
    miss = 0;
    for (int j = 0; j < 3; j++) {
      cin >> dec;
      if (dec == 1)
        hit++;
      else
        miss++;
    }
    if (hit > 1)
      flag++;
  }
  cout << flag;
  return 0;
}
