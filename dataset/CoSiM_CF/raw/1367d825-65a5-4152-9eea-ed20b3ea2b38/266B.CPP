#include <cstring>
#include <ctype.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

int main() {
  int n;
  int t;
  cin >> n;
  cin >> t;
  char temp;
  char array[100];
  cin >> array;
  while (t--) {
    for (int i = 0; i < n; i++) {
      if (array[i] == 'B' && array[i + 1] == 'G') {
        array[i] = 'G';
        array[i + 1] = 'B';
        i++;
      }
    }
  }
  cout << array;
}
