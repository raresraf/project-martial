#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;
int main() {

  int n, a, b, c;
  int count = 0;
  cin >> n;
  while (n--) {
    cin >> a >> b >> c;
    if ((a + b + c) >= 2) {
      count++;
    }
  }
  cout << count;

  return 0;
}